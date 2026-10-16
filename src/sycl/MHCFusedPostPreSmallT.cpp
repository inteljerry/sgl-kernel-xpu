#include <ATen/ATen.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"
#include "sgl_kernel_ops.h"

// Small-T path of mhc_fused_post_pre (decode batches of at most 8 tokens).
//
// Same outputs as mhc_fused_post_pre: hc_post of one sublayer (new stream
// r'_j = post_j * x + sum_i comb[i][j] * r_i, stored as bf16), then hc_pre of the
// next sublayer on the bf16 streams (24 mixes from fn [24, HC * D] scaled by the
// RMS of the flattened streams, sigmoid pre/post, 20-round Sinkhorn comb, and
// layer_input = sum_k pre_k * r'_k with the optional RMSNorm).
//
// At T = 1 the existing op runs its post/mix kernel on 8 work-groups (8 of the 32
// Xe-cores stream the 1.57 MB fp32 fn) and its pre kernel on a single 96-item
// work-group per token. This path uses two kernels:
//
// 1. PostMix: one work-group per 64-wide hidden slice (D / 64 groups at every T)
//    handles all T tokens. It writes r' as bf16 and keeps the rounded values in
//    SLM, then reads its fn slice once and applies it to every token. It stores
//    per-slice partials of the 24 mixes and of the 10 unique entries of the Gram
//    matrix G_jk = <r'_j, r'_k>.
// 2. PreApply: per token, D / 256 work-groups each reduce the partials in a fixed
//    order, derive pre and write a 256-wide slice of layer_input. The RMSNorm needs
//    ||y||^2 for y = sum_k pre_k r'_k, which equals pre^T G pre, so no slice waits
//    for another. One extra work-group per token computes post and the Sinkhorn,
//    so the Sinkhorn of every token runs in parallel with the layer_input slices.
//
//    pre^T G pre loses relative accuracy when the weighted streams cancel (||y||
//    small against sum_k pre_k ||r'_k||). Below 1/64 of that bound every y slice of
//    the token recomputes ||y||^2 directly from the streams, as the existing kernel
//    does, in one fixed order; above it the fp32 error of pre^T G pre is at most
//    about 64x the rounding error of the Gram sums.
//
// There are no atomics and no device-side counters: results are deterministic and
// both kernels can be captured in an XPU graph.

namespace {

constexpr int HC = 4;
constexpr int HC2 = HC * HC;
constexpr int HC3 = (2 + HC) * HC;         // 24 mixes: pre, post, comb
constexpr int N_GRAM = HC * (HC + 1) / 2;  // unique entries of the 4 x 4 Gram matrix
constexpr int N_PARTIAL = HC3 + N_GRAM;    // values reduced per token and slice
constexpr int SINKHORN_ITERS = 20;

constexpr int64_t kSmallTMaxTokens = 8;

constexpr int SG_SIZE = 16;
constexpr int VEC_SIZE = 4;

// PostMix: a slice is one 4-wide vector per lane of a sub-group.
constexpr int PM_SLICE = SG_SIZE * VEC_SIZE;  // 64 hidden positions
constexpr int PM_NUM_VEC = PM_SLICE / VEC_SIZE;
constexpr int PM_NUM_SG = 8;
constexpr int PM_WG_SIZE = PM_NUM_SG * SG_SIZE;
constexpr int PM_ROWS_PER_SG = HC3 / PM_NUM_SG;  // fn rows per sub-group

static_assert(PM_ROWS_PER_SG * PM_NUM_SG == HC3, "fn rows must split evenly over sub-groups");
static_assert(kSmallTMaxTokens * PM_NUM_VEC <= PM_WG_SIZE, "one post item per work-item");
static_assert(kSmallTMaxTokens <= PM_NUM_SG, "one Gram token per sub-group");

// PreApply: a slice is one 4-wide vector per work-item.
constexpr int PA_NUM_SG = 4;
constexpr int PA_WG_SIZE = PA_NUM_SG * SG_SIZE;
constexpr int PA_SLICE = PA_WG_SIZE * VEC_SIZE;  // 256 hidden positions

// ||y||^2 from the Gram matrix is used while it is at least this fraction of
// (sum_k pre_k ||r'_k||)^2; below it ||y||^2 is summed directly.
constexpr float kGramCancelLimit = 1.0f / 64.0f;

// Index of G_jk (j <= k) in the packed upper triangle:
// (0,0) (0,1) (0,2) (0,3) (1,1) (1,2) (1,3) (2,2) (2,3) (3,3).
constexpr int gram_index(int j, int k) {
  return j * HC - j * (j - 1) / 2 + (k - j);
}

template <typename T>
using global_cptr = sycl::multi_ptr<const T, sycl::access::address_space::global_space>;
template <typename T>
using global_ptr = sycl::multi_ptr<T, sycl::access::address_space::global_space>;

// Butterfly sum over the sub-group. Every lane ends with the same value, and the
// order of additions is fixed.
inline float mhc_sub_group_sum(sycl::sub_group sg, float v) {
  v += sycl::permute_group_by_xor(sg, v, 8);
  v += sycl::permute_group_by_xor(sg, v, 4);
  v += sycl::permute_group_by_xor(sg, v, 2);
  v += sycl::permute_group_by_xor(sg, v, 1);
  return v;
}

inline float mhc_sigmoid(float z) {
  return 1.0f / (1.0f + sycl::exp(-z));
}

struct MHCSmallTPostMixKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  using bf16_t = sycl::ext::oneapi::bfloat16;
  using vec_bf16 = vec_t<bf16_t, VEC_SIZE>;
  using vec_f32 = vec_t<float, VEC_SIZE>;

  const bf16_t* __restrict__ x_ptr;         // [T, D]
  const bf16_t* __restrict__ residual_ptr;  // [T, HC, D]
  const float* __restrict__ post_ptr;       // [T, HC]
  const float* __restrict__ comb_ptr;       // [T, HC, HC]
  const float* __restrict__ fn_ptr;         // [HC3, HC * D]
  bf16_t* __restrict__ residual_out_ptr;    // [T, HC, D]
  float* __restrict__ partial_ptr;          // [T, N_PARTIAL, n_slices]

  int T_total;
  int hidden_size;
  int n_slices;

  // bf16-rounded new streams of this slice as fp32: [kSmallTMaxTokens, HC, PM_NUM_VEC]
  sycl::local_accessor<vec_f32, 1> slm_res_;

  MHCSmallTPostMixKernel(
      const bf16_t* x,
      const bf16_t* residual,
      const float* post,
      const float* comb,
      const float* fn,
      bf16_t* residual_out,
      float* partial,
      int T_,
      int D_,
      int n_slices_)
      : x_ptr(x),
        residual_ptr(residual),
        post_ptr(post),
        comb_ptr(comb),
        fn_ptr(fn),
        residual_out_ptr(residual_out),
        partial_ptr(partial),
        T_total(T_),
        hidden_size(D_),
        n_slices(n_slices_) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm_res_ = sycl::local_accessor<vec_f32, 1>(kSmallTMaxTokens * HC * PM_NUM_VEC, cgh);
  }

  [[sycl::reqd_work_group_size(PM_WG_SIZE)]] [[sycl::reqd_sub_group_size(SG_SIZE)]] void
  operator()(sycl::nd_item<1> item) const {
    const int slice = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));

    sycl::sub_group sg = item.get_sub_group();
    const int sg_id = static_cast<int>(sg.get_group_id()[0]);
    const int lane = static_cast<int>(sg.get_local_id()[0]);

    const int h0 = slice * PM_SLICE;
    const int64_t hc_hidden = static_cast<int64_t>(HC) * hidden_size;

    // This sub-group's fn rows at the lane's 4 columns of each stream. Issued before
    // the post step so the DRAM reads overlap it.
    vec_f32 fn_reg[PM_ROWS_PER_SG][HC];
#pragma unroll
    for (int r = 0; r < PM_ROWS_PER_SG; ++r) {
      const int64_t row_base = static_cast<int64_t>(sg_id * PM_ROWS_PER_SG + r) * hc_hidden;
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        fn_reg[r][k].load(
            0, global_cptr<float>(&fn_ptr[row_base + static_cast<int64_t>(k) * hidden_size + h0 + lane * VEC_SIZE]));
      }
    }

    // hc_post: one (token, 4-wide vector) item per work-item, same arithmetic order
    // as the existing FMA kernel.
    if (tid < T_total * PM_NUM_VEC) {
      const int t = tid / PM_NUM_VEC;
      const int hv = tid - t * PM_NUM_VEC;
      const int h = h0 + hv * VEC_SIZE;

      float post_local[HC];
      float comb_local[HC * HC];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        post_local[j] = post_ptr[t * HC + j];
      }
#pragma unroll
      for (int jk = 0; jk < HC * HC; ++jk) {
        comb_local[jk] = comb_ptr[t * HC * HC + jk];
      }

      vec_bf16 x_vec;
      x_vec.load(0, global_cptr<bf16_t>(&x_ptr[static_cast<int64_t>(t) * hidden_size + h]));
      float x_f[VEC_SIZE];
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        x_f[i] = static_cast<float>(x_vec[i]);
      }

      float r_f[HC][VEC_SIZE];
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        vec_bf16 r_vec;
        r_vec.load(0, global_cptr<bf16_t>(&residual_ptr[(static_cast<int64_t>(t) * HC + k) * hidden_size + h]));
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          r_f[k][i] = static_cast<float>(r_vec[i]);
        }
      }

#pragma unroll
      for (int j = 0; j < HC; ++j) {
        vec_bf16 out_vec;
        vec_f32 out_f;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          float v = post_local[j] * x_f[i];
#pragma unroll
          for (int k = 0; k < HC; ++k) {
            v += comb_local[k * HC + j] * r_f[k][i];
          }
          out_vec[i] = static_cast<bf16_t>(v);
          out_f[i] = static_cast<float>(out_vec[i]);
        }
        out_vec.store(0, global_ptr<bf16_t>(&residual_out_ptr[(static_cast<int64_t>(t) * HC + j) * hidden_size + h]));
        slm_res_[(t * HC + j) * PM_NUM_VEC + hv] = out_f;
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    // Mixes: this sub-group's fn rows applied to every token of the slice.
    float acc[kSmallTMaxTokens][PM_ROWS_PER_SG];
#pragma unroll
    for (int t = 0; t < kSmallTMaxTokens; ++t) {
#pragma unroll
      for (int r = 0; r < PM_ROWS_PER_SG; ++r) {
        acc[t][r] = 0.0f;
      }
    }

#pragma unroll
    for (int t = 0; t < kSmallTMaxTokens; ++t) {
      if (t < T_total) {
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          const vec_f32 rv = slm_res_[(t * HC + k) * PM_NUM_VEC + lane];
#pragma unroll
          for (int r = 0; r < PM_ROWS_PER_SG; ++r) {
#pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
              acc[t][r] += fn_reg[r][k][i] * rv[i];
            }
          }
        }
      }
    }

#pragma unroll
    for (int t = 0; t < kSmallTMaxTokens; ++t) {
      if (t < T_total) {
#pragma unroll
        for (int r = 0; r < PM_ROWS_PER_SG; ++r) {
          const float v = mhc_sub_group_sum(sg, acc[t][r]);
          if (lane == 0) {
            const int o = sg_id * PM_ROWS_PER_SG + r;
            partial_ptr[(static_cast<int64_t>(t) * N_PARTIAL + o) * n_slices + slice] = v;
          }
        }
      }
    }

    // Gram matrix of token sg_id over this slice.
    if (sg_id < T_total) {
      const int t = sg_id;
      vec_f32 rv[HC];
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        rv[k] = slm_res_[(t * HC + k) * PM_NUM_VEC + lane];
      }
      float* dst = partial_ptr + (static_cast<int64_t>(t) * N_PARTIAL + HC3) * n_slices + slice;
#pragma unroll
      for (int j = 0; j < HC; ++j) {
#pragma unroll
        for (int k = j; k < HC; ++k) {
          float g = 0.0f;
#pragma unroll
          for (int i = 0; i < VEC_SIZE; ++i) {
            g += rv[j][i] * rv[k][i];
          }
          g = mhc_sub_group_sum(sg, g);
          if (lane == 0) {
            dst[static_cast<int64_t>(gram_index(j, k)) * n_slices] = g;
          }
        }
      }
    }
  }
};

template <bool HAS_NORM>
struct MHCSmallTPreApplyKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  using bf16_t = sycl::ext::oneapi::bfloat16;
  using vec_bf16 = vec_t<bf16_t, VEC_SIZE>;

  // Values a work-group reduces: y-slices need the pre mixes and the Gram matrix,
  // the comb work-group needs the post and comb mixes and the Gram diagonal.
  static constexpr int N_Y_VALUES = HC + N_GRAM;
  static constexpr int N_COMB_VALUES = HC + HC2 + HC;

  const float* __restrict__ partial_ptr;    // [T, N_PARTIAL, n_slices]
  const float* __restrict__ hc_scale;       // [3]
  const float* __restrict__ hc_base;        // [HC3]
  const bf16_t* __restrict__ residual_ptr;  // [T, HC, D] (new streams)
  const bf16_t* __restrict__ norm_weight;   // [D], unused without norm
  float* __restrict__ post_mix;             // [T, HC]
  float* __restrict__ comb_mix;             // [T, HC2]
  bf16_t* __restrict__ layer_input;         // [T, D]

  int T_total;
  int hidden_size;
  int n_slices;
  int n_blocks;  // y-slices per token; block n_blocks is the comb work-group
  float rms_eps;
  float hc_pre_eps;
  float hc_sinkhorn_eps;
  float hc_post_mult_value;
  float norm_eps;

  // Reduced totals, indexed like the partials.
  sycl::local_accessor<float, 1> slm_;

  MHCSmallTPreApplyKernel(
      const float* partial,
      const float* hc_scale_,
      const float* hc_base_,
      const bf16_t* residual,
      const bf16_t* norm_weight_,
      float* post_mix_,
      float* comb_mix_,
      bf16_t* layer_input_,
      int T_,
      int D_,
      int n_slices_,
      int n_blocks_,
      float rms_eps_,
      float hc_pre_eps_,
      float hc_sinkhorn_eps_,
      float hc_post_mult_value_,
      float norm_eps_)
      : partial_ptr(partial),
        hc_scale(hc_scale_),
        hc_base(hc_base_),
        residual_ptr(residual),
        norm_weight(norm_weight_),
        post_mix(post_mix_),
        comb_mix(comb_mix_),
        layer_input(layer_input_),
        T_total(T_),
        hidden_size(D_),
        n_slices(n_slices_),
        n_blocks(n_blocks_),
        rms_eps(rms_eps_),
        hc_pre_eps(hc_pre_eps_),
        hc_sinkhorn_eps(hc_sinkhorn_eps_),
        hc_post_mult_value(hc_post_mult_value_),
        norm_eps(norm_eps_) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm_ = sycl::local_accessor<float, 1>(N_PARTIAL + PA_NUM_SG, cgh);
  }

  static inline int y_value(int i) {
    return i < HC ? i : HC3 + (i - HC);
  }

  static inline int comb_value(int i) {
    if (i < HC + HC2) return HC + i;
    const int j = i - HC - HC2;
    return HC3 + gram_index(j, j);
  }

  // ||y||^2 for y = sum_k pre_k r'_k summed over the whole hidden size: per
  // work-item over a fixed stride, then the sub-group butterfly, then the
  // sub-groups in order. Every work-group of the token computes the same value.
  // Called by all work-items of the work-group (it has a barrier).
  inline float direct_y_sqr(sycl::nd_item<1> item, sycl::sub_group sg, int t, const float* pre) const {
    const int tid = static_cast<int>(item.get_local_id(0));
    const int sg_id = static_cast<int>(sg.get_group_id()[0]);
    const int lane = static_cast<int>(sg.get_local_id()[0]);
    float s = 0.0f;
    for (int h = tid * VEC_SIZE; h < hidden_size; h += PA_SLICE) {
      float accum[VEC_SIZE] = {};
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        vec_bf16 r_vec;
        r_vec.load(0, global_cptr<bf16_t>(&residual_ptr[(static_cast<int64_t>(t) * HC + k) * hidden_size + h]));
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          accum[i] += pre[k] * static_cast<float>(r_vec[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        s += accum[i] * accum[i];
      }
    }
    s = mhc_sub_group_sum(sg, s);
    if (lane == 0) {
      slm_[N_PARTIAL + sg_id] = s;
    }
    item.barrier(sycl::access::fence_space::local_space);
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < PA_NUM_SG; ++i) {
      total += slm_[N_PARTIAL + i];
    }
    return total;
  }

  // RMS scale of the flattened [HC * D] streams; sum of squares = trace(G).
  inline float stream_rms_scale() const {
    const float sqrsum = slm_[HC3 + gram_index(0, 0)] + slm_[HC3 + gram_index(1, 1)] + slm_[HC3 + gram_index(2, 2)] +
                         slm_[HC3 + gram_index(3, 3)];
    return sycl::rsqrt(sqrsum / static_cast<float>(HC * hidden_size) + rms_eps);
  }

  inline void post_and_sinkhorn(sycl::sub_group sg, int t, int lane, float rms) const {
    if (lane < HC) {
      const float post_logit = slm_[HC + lane] * rms * hc_scale[1] + hc_base[HC + lane];
      post_mix[static_cast<int64_t>(t) * HC + lane] = hc_post_mult_value * mhc_sigmoid(post_logit);
    }

    // Lane l holds comb[l / 4][l % 4]: xor 1, 2 walk a row, xor 4, 8 a column.
    const float comb_logit = slm_[2 * HC + lane] * rms * hc_scale[2] + hc_base[2 * HC + lane];

    float row_max = comb_logit;
#pragma unroll
    for (int mask = 1; mask < HC; mask <<= 1)
      row_max = sycl::fmax(row_max, sycl::permute_group_by_xor(sg, row_max, mask));

    float comb_val = sycl::exp(comb_logit - row_max);
    float row_sum = comb_val;
#pragma unroll
    for (int mask = 1; mask < HC; mask <<= 1)
      row_sum += sycl::permute_group_by_xor(sg, row_sum, mask);
    comb_val = comb_val / row_sum + hc_sinkhorn_eps;

    float col_sum = comb_val;
#pragma unroll
    for (int mask = HC; mask < HC2; mask <<= 1)
      col_sum += sycl::permute_group_by_xor(sg, col_sum, mask);
    comb_val = comb_val / (col_sum + hc_sinkhorn_eps);

#pragma unroll
    for (int iter = 1; iter < SINKHORN_ITERS; ++iter) {
      row_sum = comb_val;
#pragma unroll
      for (int mask = 1; mask < HC; mask <<= 1)
        row_sum += sycl::permute_group_by_xor(sg, row_sum, mask);
      comb_val = comb_val / (row_sum + hc_sinkhorn_eps);

      col_sum = comb_val;
#pragma unroll
      for (int mask = HC; mask < HC2; mask <<= 1)
        col_sum += sycl::permute_group_by_xor(sg, col_sum, mask);
      comb_val = comb_val / (col_sum + hc_sinkhorn_eps);
    }

    comb_mix[static_cast<int64_t>(t) * HC2 + lane] = comb_val;
  }

  [[sycl::reqd_work_group_size(PA_WG_SIZE)]] [[sycl::reqd_sub_group_size(SG_SIZE)]] void
  operator()(sycl::nd_item<1> item) const {
    const int group = static_cast<int>(item.get_group(0));
    const int t = group / (n_blocks + 1);
    const int blk = group - t * (n_blocks + 1);
    const bool comb_block = blk == n_blocks;
    const int tid = static_cast<int>(item.get_local_id(0));

    sycl::sub_group sg = item.get_sub_group();
    const int sg_id = static_cast<int>(sg.get_group_id()[0]);
    const int lane = static_cast<int>(sg.get_local_id()[0]);

    // Second-stage reduce over the slices, in the same order in every work-group.
    const int n_values = comb_block ? N_COMB_VALUES : N_Y_VALUES;
    for (int i = sg_id; i < n_values; i += PA_NUM_SG) {
      const int v = comb_block ? comb_value(i) : y_value(i);
      const float* p = partial_ptr + (static_cast<int64_t>(t) * N_PARTIAL + v) * n_slices;
      float s = 0.0f;
      for (int sl = lane; sl < n_slices; sl += SG_SIZE) {
        s += p[sl];
      }
      s = mhc_sub_group_sum(sg, s);
      if (lane == 0) {
        slm_[v] = s;
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    const float rms = stream_rms_scale();

    if (comb_block) {
      if (sg_id == 0) {
        post_and_sinkhorn(sg, t, lane, rms);
      }
      return;
    }

    float pre[HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      const float pre_logit = slm_[j] * rms * hc_scale[0] + hc_base[j];
      pre[j] = mhc_sigmoid(pre_logit) + hc_pre_eps;
    }

    // RMSNorm scale of y = sum_k pre_k r'_k: ||y||^2 = pre^T G pre, unless the
    // streams cancel (see kGramCancelLimit). The branch depends only on values
    // every work-item of the token reads from SLM, so it is uniform per work-group.
    float y_scale = 1.0f;
    if constexpr (HAS_NORM) {
      float y_sqr = 0.0f;
      float bound = 0.0f;
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        bound += pre[j] * sycl::sqrt(slm_[HC3 + gram_index(j, j)]);
#pragma unroll
        for (int k = j; k < HC; ++k) {
          const float w = (j == k) ? 1.0f : 2.0f;
          y_sqr += w * pre[j] * pre[k] * slm_[HC3 + gram_index(j, k)];
        }
      }
      if (y_sqr < kGramCancelLimit * bound * bound) {
        y_sqr = direct_y_sqr(item, sg, t, pre);
      }
      y_scale = sycl::rsqrt(sycl::fmax(y_sqr, 0.0f) / static_cast<float>(hidden_size) + norm_eps);
    }

    const int h = blk * PA_SLICE + tid * VEC_SIZE;
    float accum[VEC_SIZE] = {};
#pragma unroll
    for (int k = 0; k < HC; ++k) {
      vec_bf16 r_vec;
      r_vec.load(0, global_cptr<bf16_t>(&residual_ptr[(static_cast<int64_t>(t) * HC + k) * hidden_size + h]));
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        accum[i] += pre[k] * static_cast<float>(r_vec[i]);
      }
    }

    vec_bf16 out;
    if constexpr (HAS_NORM) {
      vec_bf16 w_vec;
      w_vec.load(0, global_cptr<bf16_t>(&norm_weight[h]));
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        out[i] = static_cast<bf16_t>(accum[i] * y_scale * static_cast<float>(w_vec[i]));
      }
    } else {
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        out[i] = static_cast<bf16_t>(accum[i]);
      }
    }
    out.store(0, global_ptr<bf16_t>(&layer_input[static_cast<int64_t>(t) * hidden_size + h]));
  }
};

inline bool is_aligned(const at::Tensor& t, uintptr_t bytes) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes == 0;
}

template <bool HAS_NORM>
void launch_mhc_small_t_pre_apply(
    sycl::queue& q,
    const at::Tensor& partial,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    const at::Tensor& residual_cur,
    const std::optional<at::Tensor>& norm_weight,
    at::Tensor& post_mix,
    at::Tensor& comb_mix,
    at::Tensor& layer_input,
    int64_t T,
    int64_t hidden_size,
    int64_t n_slices,
    double rms_eps,
    double hc_pre_eps,
    double hc_sinkhorn_eps,
    double hc_post_mult_value,
    double norm_eps) {
  using bf16_t = sycl::ext::oneapi::bfloat16;
  const int64_t n_blocks = hidden_size / PA_SLICE;
  const bf16_t* norm_weight_ptr =
      HAS_NORM ? reinterpret_cast<const bf16_t*>(norm_weight.value().data_ptr<at::BFloat16>()) : nullptr;

  MHCSmallTPreApplyKernel<HAS_NORM> ker(
      partial.data_ptr<float>(),
      hc_scale.data_ptr<float>(),
      hc_base.data_ptr<float>(),
      reinterpret_cast<const bf16_t*>(residual_cur.data_ptr<at::BFloat16>()),
      norm_weight_ptr,
      post_mix.data_ptr<float>(),
      comb_mix.data_ptr<float>(),
      reinterpret_cast<bf16_t*>(layer_input.data_ptr<at::BFloat16>()),
      static_cast<int>(T),
      static_cast<int>(hidden_size),
      static_cast<int>(n_slices),
      static_cast<int>(n_blocks),
      static_cast<float>(rms_eps),
      static_cast<float>(hc_pre_eps),
      static_cast<float>(hc_sinkhorn_eps),
      static_cast<float>(hc_post_mult_value),
      static_cast<float>(norm_eps));
  sycl_kernel_submit(T * (n_blocks + 1) * PA_WG_SIZE, static_cast<int64_t>(PA_WG_SIZE), q, ker);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> SGL_KERNEL_EXPORT mhc_fused_post_pre_small_t(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix,
    const at::Tensor& fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    double rms_eps,
    double hc_pre_eps,
    double hc_sinkhorn_eps,
    double hc_post_mult_value,
    int64_t sinkhorn_repeat,
    int64_t n_splits,
    std::optional<at::Tensor> norm_weight,
    std::optional<double> norm_eps) {
  TORCH_CHECK(x.dim() == 2, "x must be 2D [T, D]");
  TORCH_CHECK(residual.dim() == 3, "residual must be 3D [T, HC, D]");

  const int64_t num_tokens = x.size(0);
  const int64_t hidden_size = x.size(1);

  // The kernels load x, residual and norm_weight as 8-byte and fn as 16-byte vectors.
  const bool aligned = is_aligned(x, 8) && is_aligned(residual, 8) && is_aligned(fn, 16) &&
                       (!norm_weight.has_value() || is_aligned(norm_weight.value(), 8));

  // Outside the small-T shapes the existing op runs unchanged; n_splits only
  // applies there, the small-T path always uses hidden_size / 64 slices.
  if (num_tokens == 0 || num_tokens > kSmallTMaxTokens || hidden_size % PA_SLICE != 0 || !aligned) {
    return mhc_fused_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        norm_weight,
        norm_eps);
  }

  CHECK_INPUT(x);
  CHECK_INPUT(residual);
  CHECK_INPUT(post_layer_mix);
  CHECK_INPUT(comb_res_mix);
  CHECK_INPUT(fn);
  CHECK_INPUT(hc_scale);
  CHECK_INPUT(hc_base);

  const int64_t hc_mult = residual.size(1);
  TORCH_CHECK(hc_mult == HC, "mhc_fused_post_pre_small_t currently supports only HC=4");
  TORCH_CHECK(
      residual.size(0) == num_tokens && residual.size(2) == hidden_size,
      "residual must be [T, HC, D] matching x [T, D]");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
  TORCH_CHECK(post_layer_mix.scalar_type() == at::kFloat, "post_layer_mix must be float32");
  TORCH_CHECK(comb_res_mix.scalar_type() == at::kFloat, "comb_res_mix must be float32");
  TORCH_CHECK(fn.scalar_type() == at::kFloat, "fn must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat, "hc_scale must be float32");
  TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_base must be float32");
  TORCH_CHECK(
      fn.dim() == 2 && fn.size(0) == HC3 && fn.size(1) == HC * hidden_size,
      "fn must be [",
      HC3,
      ", ",
      HC * hidden_size,
      "], got ",
      fn.sizes());
  TORCH_CHECK(hc_scale.numel() == 3, "hc_scale must have 3 elements");
  TORCH_CHECK(hc_base.numel() == HC3, "hc_base must have ", HC3, " elements");
  TORCH_CHECK(sinkhorn_repeat == SINKHORN_ITERS, "sinkhorn_repeat must be ", SINKHORN_ITERS, ", got ", sinkhorn_repeat);

  at::Tensor post_2d = post_layer_mix;
  if (post_2d.dim() == 3) {
    TORCH_CHECK(post_2d.size(2) == 1, "post_layer_mix last dim must be 1 when rank=3");
    post_2d = post_2d.squeeze(-1);
  }
  TORCH_CHECK(post_2d.dim() == 2, "post_layer_mix must be [T, HC] or [T, HC, 1]");
  TORCH_CHECK(post_2d.size(0) == num_tokens && post_2d.size(1) == hc_mult, "post_layer_mix shape mismatch");

  at::Tensor comb_3d = comb_res_mix;
  if (comb_3d.dim() == 2) {
    TORCH_CHECK(comb_3d.size(1) == hc_mult * hc_mult, "comb_res_mix rank-2 shape mismatch");
    comb_3d = comb_3d.view({num_tokens, hc_mult, hc_mult});
  }
  TORCH_CHECK(comb_3d.dim() == 3, "comb_res_mix must be [T, HC, HC] or [T, HC*HC]");
  TORCH_CHECK(
      comb_3d.size(0) == num_tokens && comb_3d.size(1) == hc_mult && comb_3d.size(2) == hc_mult,
      "comb_res_mix shape mismatch");

  if (norm_weight.has_value()) {
    CHECK_INPUT(norm_weight.value());
    TORCH_CHECK(norm_weight.value().scalar_type() == at::kBFloat16, "norm_weight must be bfloat16");
    TORCH_CHECK(norm_weight.value().numel() == hidden_size, "norm_weight size must match hidden_size");
  }

  using bf16_t = sycl::ext::oneapi::bfloat16;
  const int64_t n_slices = hidden_size / PM_SLICE;

  at::Tensor residual_cur = at::empty_like(residual);
  at::Tensor partial = at::empty({num_tokens, N_PARTIAL, n_slices}, residual.options().dtype(at::kFloat));
  at::Tensor post_mix_cur = at::empty({num_tokens, hc_mult}, residual.options().dtype(at::kFloat));
  at::Tensor comb_mix_cur = at::empty({num_tokens, hc_mult, hc_mult}, residual.options().dtype(at::kFloat));
  at::Tensor layer_input_cur = at::empty({num_tokens, hidden_size}, x.options().dtype(at::kBFloat16));

  auto q = dpcppGetCurrentQueue();

  MHCSmallTPostMixKernel post_mix_ker(
      reinterpret_cast<const bf16_t*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<const bf16_t*>(residual.data_ptr<at::BFloat16>()),
      post_2d.data_ptr<float>(),
      comb_3d.data_ptr<float>(),
      fn.data_ptr<float>(),
      reinterpret_cast<bf16_t*>(residual_cur.data_ptr<at::BFloat16>()),
      partial.data_ptr<float>(),
      static_cast<int>(num_tokens),
      static_cast<int>(hidden_size),
      static_cast<int>(n_slices));
  sycl_kernel_submit(n_slices * PM_WG_SIZE, static_cast<int64_t>(PM_WG_SIZE), q, post_mix_ker);

  if (norm_weight.has_value()) {
    launch_mhc_small_t_pre_apply<true>(
        q,
        partial,
        hc_scale,
        hc_base,
        residual_cur,
        norm_weight,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
        num_tokens,
        hidden_size,
        n_slices,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        norm_eps.value_or(1e-6));
  } else {
    launch_mhc_small_t_pre_apply<false>(
        q,
        partial,
        hc_scale,
        hc_base,
        residual_cur,
        norm_weight,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
        num_tokens,
        hidden_size,
        n_slices,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        0.0);
  }

  return {residual_cur, post_mix_cur.unsqueeze(-1), comb_mix_cur, layer_input_cur};
}
