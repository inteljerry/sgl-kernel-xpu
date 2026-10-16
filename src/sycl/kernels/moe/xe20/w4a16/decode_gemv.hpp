// Decode-only MXFP4 W4A16 routed-expert MoE as two GEMV kernels.
//
// Computes what fused_experts(use_mxfp4_w4a16=True, activation="silu",
// swiglu_limit=L, routed_scaling_factor=rsf) computes, for small decode
// batches (T * topk <= 64 routes), with the same bf16 rounding points:
//
//   h      = bf16(x[t] @ w13[e]^T)                    [2I]  (gate = h[:I], up = h[I:])
//   a      = bf16(silu(bf16(min(gate, L))) * bf16(clamp(up, -L, L)))   [I]
//   y      = bf16(a @ w2[e]^T)                        [H]
//   out[t] = bf16(sum_k y_k * w_k * rsf)              (k in topk order, fp32)
//
// It consumes the weight tensors of fused_experts as they are, with no re-layout:
//   w13       [E, 2I, H/2]  int8/uint8, two E2M1 codes per byte (low nibble first),
//                           gate rows [0, I) then up rows [I, 2I)
//   w13_scale [E, 2I, H/32] uint8/float8_e8m0fnu, one E8M0 exponent per 32 inputs
//   w2        [E, H, I/2], w2_scale [E, H, I/32], same encoding
//
// Kernel 1 (GEMM1 + clamped SwiGLU): one work-group per (route, block of gate
// rows). The first route of each distinct expert leads it; the other routes of that
// expert exit before any load. The leader streams the expert's gate and up rows of
// its block once per pair of routed tokens, so the weight stream covers only active
// experts, and an expert chosen by one or two tokens is read once. No activation
// scatter, expert walk or atomic counter is needed.
//
// Kernel 2 (GEMM2 + combine): one work-group per (token, block of output rows); it
// loops over the token's topk experts in topk order and accumulates the weighted,
// bf16-rounded expert outputs in fp32, so the combine needs no atomics and its
// summation order is fixed. An expert shared by several tokens is read once per
// token. Adjacent work-groups process the tokens of one row block, which can
// permit cache reuse.
//
// Loads: every weight load is a 16-byte vector per lane with consecutive lanes on
// consecutive addresses, so one sub-group message covers 256 contiguous bytes (four
// full 64-byte lines): part of one row in GEMM1, and one row (I = 512) or two rows
// (I = 256) in GEMM2. One lane owns exactly one 32-value scale group.
//
// Dequantization: an E2M1 nibble moved to bits 31..28, arithmetically shifted right
// by 6 and masked with 0x81C00000 is the fp32 value code * 2^-126 (the 0.5 code
// becomes the fp32 denormal 2^-127, which the -no-ftz build keeps). Activations are
// pre-scaled by 2^64 so every product is a normal fp32 value; each 32-value group
// sum is multiplied by 2^62 (= 2^126 / 2^64, exact) and then once by its E8M0
// scale, so the scale order cannot underflow before the true magnitude is restored.
// This needs |x| < 2^64 for every activation, far above bf16 hidden states.
//
// Graph safety: no host synchronization, no device-side counters, a grid that
// depends only on tensor shapes, and no data-dependent host control flow.

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace moe_w4a16_gemv {

using bf16_t = sycl::ext::oneapi::bfloat16;

static constexpr int SG_SIZE = 16;
// Sub-groups per work-group. The sub-groups of one work-group share no data.
static constexpr int SG_PER_WG = 4;
static constexpr int WG_SIZE = SG_SIZE * SG_PER_WG;
// Each lane owns one 32-value scale group per row message: 16 packed bytes.
static constexpr int GROUP = 32;
static constexpr int LANE_BYTES = 16;
// Values of one row covered by one sub-group message along K.
static constexpr int SG_K_VALUES = SG_SIZE * GROUP;
// Routes are checked 16 per sub-group step; 4 steps cover 64 routes.
static constexpr int ROUTE_STEPS = 4;
static constexpr int MAX_ROUTES = SG_SIZE * ROUTE_STEPS;

static constexpr float X_PRESCALE = 0x1p64f;
static constexpr float DOT_RESCALE = 0x1p62f;
// Same constant as SiluAndMulClamp.cpp.
static constexpr float LOG2E = 1.442695040888963f;

// E2M1 nibble j of w (element j, low nibble first) as fp32, times 2^-126.
inline float e2m1_scaled(uint32_t w, int j) {
  const int32_t v = static_cast<int32_t>(w << (28 - 4 * j)) >> 6;
  return sycl::bit_cast<float>(static_cast<uint32_t>(v) & 0x81C00000u);
}

// E8M0 exponent byte as 2^(e - 127); e = 0 gives 0 and e = 255 gives +inf, as in
// the existing grouped GEMM.
inline float e8m0_to_float(uint32_t e) {
  return sycl::bit_cast<float>(e << 23);
}

inline float bf16_lo(uint32_t p) {
  return sycl::bit_cast<float>(p << 16);
}

inline float bf16_hi(uint32_t p) {
  return sycl::bit_cast<float>(p & 0xFFFF0000u);
}

// Eight bf16 activations (four packed dwords) as fp32, pre-scaled by 2^64.
inline void unpack_x8(const sycl::uint4& p, float (&xs)[8]) {
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    xs[2 * m] = bf16_lo(p[m]) * X_PRESCALE;
    xs[2 * m + 1] = bf16_hi(p[m]) * X_PRESCALE;
  }
}

// acc + dot product of the eight E2M1 codes in w with eight pre-scaled
// activations, accumulated in element order.
inline float dot8(uint32_t w, const float (&xs)[8], float acc) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    acc = sycl::fma(e2m1_scaled(w, j), xs[j], acc);
  }
  return acc;
}

// Dot product of one lane's 32-value group (four packed dwords of codes) with its
// 32 pre-scaled activations, rescaled to the true magnitude and multiplied by the
// group's E8M0 scale.
inline float group_dot(const sycl::uint4& w, const sycl::uint4 (&x)[4], uint32_t scale) {
  float g = 0.0f;
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    float xs[8];
    unpack_x8(x[d], xs);
    g = dot8(w[d], xs, g);
  }
  return (g * DOT_RESCALE) * e8m0_to_float(scale);
}

inline float round_bf16(float v) {
  return static_cast<float>(static_cast<bf16_t>(v));
}

// Same arithmetic as silu_and_mul_clamp_elem in SiluAndMulClamp.cpp, applied to
// gate and up values that are already bf16-rounded.
inline float silu_and_mul_clamp(float gate, float up, float limit_bf16) {
  const float gate_c = round_bf16(sycl::fmin(gate, limit_bf16));
  const float up_c = round_bf16(sycl::fmax(-limit_bf16, sycl::fmin(up, limit_bf16)));
  const float sig = sycl::native::recip(1.0f + sycl::native::exp2(-gate_c * LOG2E));
  return gate_c * sig * up_c;
}

// Butterfly sum over WIDTH consecutive lanes; every lane of a segment ends up with
// the same value because each step adds the same two operands.
template <int WIDTH>
inline float segment_sum(sycl::sub_group sg, float v) {
#pragma unroll
  for (int o = WIDTH / 2; o >= 1; o /= 2) {
    v += sycl::permute_group_by_xor(sg, v, o);
  }
  return v;
}

// ---------------------------------------------------------------------------
// Kernel 1: GEMM1 over active experts with the clamped SwiGLU epilogue.
//
// Grid: num_routes * (I / (GATE_ROWS * SG_PER_WG)) work-groups of WG_SIZE.
// Work-group wg serves route r = wg % num_routes and gate-row block
// wg / num_routes. Each sub-group owns GATE_ROWS gate rows and the matching up rows
// (the existing layout keeps gate in rows [0, I) and up in rows [I, 2I)).
// ---------------------------------------------------------------------------
template <typename index_t>
struct MoEW4A16GemvUpKernel {
  static constexpr int GATE_ROWS = 2;
  static constexpr int ROWS = 2 * GATE_ROWS;

  // One K step of the weight stream. For a single token its activations ride in
  // the stage too, so they are prefetched with the weights; for two tokens they
  // are loaded at compute time.
  template <int NT>
  struct Stage {
    sycl::uint4 w[ROWS];
    uint32_t s[ROWS];
    sycl::uint4 x[NT == 1 ? 4 : 1];
  };

  template <int NT>
  struct Pass {
    const uint8_t* w[ROWS];
    const uint8_t* s[ROWS];
    const uint32_t* x[NT];
  };

  const uint32_t* x_;         // [T, H/2] bf16 pairs
  const uint8_t* w13_;        // [E, 2I, H/2]
  const uint8_t* w13_scale_;  // [E, 2I, H/32]
  const index_t* topk_ids_;   // [T, topk]
  bf16_t* intermediate_;      // [T * topk, I]
  int num_experts_;
  int inter_size_;  // I
  int hidden_;      // H
  int topk_;
  int num_routes_;
  float limit_bf16_;

  MoEW4A16GemvUpKernel(
      const uint32_t* x,
      const uint8_t* w13,
      const uint8_t* w13_scale,
      const index_t* topk_ids,
      bf16_t* intermediate,
      int num_experts,
      int inter_size,
      int hidden,
      int topk,
      int num_routes,
      float limit_bf16)
      : x_(x),
        w13_(w13),
        w13_scale_(w13_scale),
        topk_ids_(topk_ids),
        intermediate_(intermediate),
        num_experts_(num_experts),
        inter_size_(inter_size),
        hidden_(hidden),
        topk_(topk),
        num_routes_(num_routes),
        limit_bf16_(limit_bf16) {}

  static inline void load_x(sycl::uint4 (&x)[4], const uint32_t* xrow, int i) {
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      x[d] = *reinterpret_cast<const sycl::uint4*>(xrow + i * (SG_K_VALUES / 2) + 4 * d);
    }
  }

  template <int NT>
  inline void load_stage(Stage<NT>& st, const Pass<NT>& p, int i) const {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      st.w[r] = *reinterpret_cast<const sycl::uint4*>(p.w[r] + i * (SG_SIZE * LANE_BYTES));
      st.s[r] = p.s[r][i * SG_SIZE];
    }
    if constexpr (NT == 1) {
      load_x(st.x, p.x[0], i);
    }
  }

  template <int NT>
  inline void compute_stage(const Stage<NT>& st, const Pass<NT>& p, int i, float (&acc)[NT][ROWS]) const {
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      sycl::uint4 x[4];
      if constexpr (NT == 1) {
#pragma unroll
        for (int d = 0; d < 4; ++d) {
          x[d] = st.x[d];
        }
      } else {
        load_x(x, p.x[n], i);
      }
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        acc[n][r] += group_dot(st.w[r], x, st.s[r]);
      }
    }
  }

  // One pass over this sub-group's rows of expert e, computing the activated values
  // of the NT routes in q (tokens q / topk).
  template <int NT>
  inline void expert_pass(sycl::sub_group sg, int lane, int e, const int (&q)[NT], int n0) const {
    const int64_t row_bytes = hidden_ / 2;
    const int64_t row_groups = hidden_ / GROUP;
    const int64_t expert_row0 = static_cast<int64_t>(e) * 2 * inter_size_;

    Pass<NT> p;
#pragma unroll
    for (int g = 0; g < GATE_ROWS; ++g) {
      const int64_t gate_row = expert_row0 + n0 + g;
      const int64_t up_row = gate_row + inter_size_;
      p.w[g] = w13_ + gate_row * row_bytes + lane * LANE_BYTES;
      p.w[GATE_ROWS + g] = w13_ + up_row * row_bytes + lane * LANE_BYTES;
      p.s[g] = w13_scale_ + gate_row * row_groups + lane;
      p.s[GATE_ROWS + g] = w13_scale_ + up_row * row_groups + lane;
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      p.x[n] = x_ + static_cast<int64_t>(q[n] / topk_) * (hidden_ / 2) + lane * (GROUP / 2);
    }

    float acc[NT][ROWS];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        acc[n][r] = 0.0f;
      }
    }

    // Two named stages: the loads of step i + 1 are in flight while step i computes.
    const int steps = hidden_ / SG_K_VALUES;
    Stage<NT> a, b;
    load_stage(a, p, 0);
    for (int i = 0; i < steps; i += 2) {
      if (i + 1 < steps) load_stage(b, p, i + 1);
      compute_stage(a, p, i, acc);
      if (i + 1 < steps) {
        if (i + 2 < steps) load_stage(a, p, i + 2);
        compute_stage(b, p, i + 1, acc);
      }
    }

#pragma unroll
    for (int n = 0; n < NT; ++n) {
      float act[GATE_ROWS];
#pragma unroll
      for (int g = 0; g < GATE_ROWS; ++g) {
        const float gate = round_bf16(segment_sum<SG_SIZE>(sg, acc[n][g]));
        const float up = round_bf16(segment_sum<SG_SIZE>(sg, acc[n][GATE_ROWS + g]));
        act[g] = silu_and_mul_clamp(gate, up, limit_bf16_);
      }
      if (lane < GATE_ROWS) {
        float v = act[0];
#pragma unroll
        for (int g = 1; g < GATE_ROWS; ++g) {
          v = (lane == g) ? act[g] : v;
        }
        intermediate_[static_cast<int64_t>(q[n]) * inter_size_ + n0 + lane] = static_cast<bf16_t>(v);
      }
    }
  }

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<1> item) const {
    const auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int wg = item.get_group(0);
    const int r = wg % num_routes_;
    const int blk = wg / num_routes_;
    const int n0 = (blk * SG_PER_WG + static_cast<int>(sg.get_group_linear_id())) * GATE_ROWS;

    const index_t id = topk_ids_[r];
    if (id < 0 || id >= num_experts_) return;
    const int e = static_cast<int>(id);

    // Route r leads expert e if no earlier route picked e. The leader collects
    // every route of e (including r) in ascending route order.
    bool earlier = false;
    uint32_t mine[ROUTE_STEPS];
#pragma unroll
    for (int j = 0; j < ROUTE_STEPS; ++j) {
      const int q = lane + SG_SIZE * j;
      const index_t eq = (q < num_routes_) ? topk_ids_[q] : index_t{-1};
      earlier = earlier || (q < r && eq == e);
      const uint32_t bit = (q >= r && eq == e) ? (1u << lane) : 0u;
      mine[j] = sycl::reduce_over_group(sg, bit, sycl::bit_or<uint32_t>());
    }
    if (sycl::any_of_group(sg, earlier)) return;

    // Two routes per weight pass; an odd last route gets a pass of its own.
    int pending = -1;
#pragma unroll
    for (int j = 0; j < ROUTE_STEPS; ++j) {
      uint32_t m = mine[j];
      while (m != 0u) {
        const int q = static_cast<int>(sycl::ctz(m)) + SG_SIZE * j;
        m &= m - 1u;
        if (pending < 0) {
          pending = q;
        } else {
          const int pair[2] = {pending, q};
          expert_pass<2>(sg, lane, e, pair, n0);
          pending = -1;
        }
      }
    }
    if (pending >= 0) {
      const int single[1] = {pending};
      expert_pass<1>(sg, lane, e, single, n0);
    }
  }
};

// ---------------------------------------------------------------------------
// Kernel 2: GEMM2 fused with the routing-weight combine.
//
// Grid: T * (H / (ROWS * SG_PER_WG)) work-groups of WG_SIZE. Work-group wg serves
// token t = wg % T and row block wg / T, so the tokens of one row block run in
// adjacent work-groups.
// LPR lanes cover one row of K = I = 32 * LPR values; one message covers 16 / LPR
// rows.
// ---------------------------------------------------------------------------
template <typename index_t, typename weight_t, int TOPK, int LPR>
struct MoEW4A16GemvDownKernel {
  static constexpr int ROWS = 4;
  static constexpr int ROWS_PER_MSG = SG_SIZE / LPR;
  static constexpr int MSGS = ROWS / ROWS_PER_MSG;
  static_assert(TOPK % 2 == 0, "TOPK must be even");
  static_assert(ROWS % ROWS_PER_MSG == 0, "ROWS must be a multiple of rows per message");

  struct Stage {
    sycl::uint4 x[4];
    sycl::uint4 w[MSGS];
    uint32_t s[MSGS];
  };

  const uint32_t* intermediate_;  // [T * topk, I/2] bf16 pairs
  const uint8_t* w2_;             // [E, H, I/2]
  const uint8_t* w2_scale_;       // [E, H, I/32]
  const index_t* topk_ids_;       // [T, topk]
  const weight_t* topk_weights_;  // [T, topk]
  bf16_t* out_;                   // [T, H]
  int num_tokens_;
  int num_experts_;
  int hidden_;
  float routed_scaling_factor_;

  MoEW4A16GemvDownKernel(
      const uint32_t* intermediate,
      const uint8_t* w2,
      const uint8_t* w2_scale,
      const index_t* topk_ids,
      const weight_t* topk_weights,
      bf16_t* out,
      int num_tokens,
      int num_experts,
      int hidden,
      float routed_scaling_factor)
      : intermediate_(intermediate),
        w2_(w2),
        w2_scale_(w2_scale),
        topk_ids_(topk_ids),
        topk_weights_(topk_weights),
        out_(out),
        num_tokens_(num_tokens),
        num_experts_(num_experts),
        hidden_(hidden),
        routed_scaling_factor_(routed_scaling_factor) {}

  inline void load_stage(Stage& st, int e, int q, int lane, int n0) const {
    if (e < 0) return;
    const int group = lane % LPR;
    const uint32_t* xrow = intermediate_ + static_cast<int64_t>(q) * (LPR * GROUP / 2) + group * (GROUP / 2);
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      st.x[d] = *reinterpret_cast<const sycl::uint4*>(xrow + 4 * d);
    }
    const int64_t row0 = static_cast<int64_t>(e) * hidden_ + n0;
#pragma unroll
    for (int m = 0; m < MSGS; ++m) {
      const int64_t row = row0 + m * ROWS_PER_MSG;
      st.w[m] = *reinterpret_cast<const sycl::uint4*>(w2_ + row * (LPR * LANE_BYTES) + lane * LANE_BYTES);
      st.s[m] = w2_scale_[row * LPR + lane];
    }
  }

  inline void compute_stage(sycl::sub_group sg, const Stage& st, int e, float wk, float (&acc)[MSGS]) const {
    if (e < 0) return;
#pragma unroll
    for (int m = 0; m < MSGS; ++m) {
      const float y = round_bf16(segment_sum<LPR>(sg, group_dot(st.w[m], st.x, st.s[m])));
      // Same expression as ApplyShuffleMulSum in MoEPrepareInputs.cpp.
      if (routed_scaling_factor_ != 1.0f) {
        acc[m] += y * wk * routed_scaling_factor_;
      } else {
        acc[m] += y * wk;
      }
    }
  }

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<1> item) const {
    const auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int wg = item.get_group(0);
    const int t = wg % num_tokens_;
    const int blk = wg / num_tokens_;
    const int n0 = (blk * SG_PER_WG + static_cast<int>(sg.get_group_linear_id())) * ROWS;

    int ek[TOPK];
    float wk[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const index_t e = topk_ids_[t * TOPK + k];
      ek[k] = (e >= 0 && e < num_experts_) ? static_cast<int>(e) : -1;
      wk[k] = static_cast<float>(topk_weights_[t * TOPK + k]);
    }

    float acc[MSGS];
#pragma unroll
    for (int m = 0; m < MSGS; ++m) {
      acc[m] = 0.0f;
    }

    Stage a, b;
    load_stage(a, ek[0], t * TOPK, lane, n0);
#pragma unroll
    for (int k = 0; k < TOPK; k += 2) {
      load_stage(b, ek[k + 1], t * TOPK + k + 1, lane, n0);
      compute_stage(sg, a, ek[k], wk[k], acc);
      if (k + 2 < TOPK) load_stage(a, ek[k + 2], t * TOPK + k + 2, lane, n0);
      compute_stage(sg, b, ek[k + 1], wk[k + 1], acc);
    }

    if (lane % LPR == 0) {
      const int seg = lane / LPR;
#pragma unroll
      for (int m = 0; m < MSGS; ++m) {
        out_[static_cast<int64_t>(t) * hidden_ + n0 + m * ROWS_PER_MSG + seg] = static_cast<bf16_t>(acc[m]);
      }
    }
  }
};

}  // namespace moe_w4a16_gemv
