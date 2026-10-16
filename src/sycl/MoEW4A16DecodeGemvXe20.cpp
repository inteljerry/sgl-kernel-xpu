// Host entry of the decode-only MXFP4 W4A16 routed-expert MoE GEMV
// (moe_w4a16_decode_gemv). The two kernels and their design notes are in
// kernels/moe/xe20/w4a16/decode_gemv.hpp.

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <initializer_list>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "kernels/moe/xe20/w4a16/decode_gemv.hpp"
#include "sgl_kernel_export.h"

namespace moe_w4a16_gemv {

template <typename index_t>
void launch_up(
    sycl::queue& queue,
    const torch::Tensor& hidden_states,
    const torch::Tensor& w13,
    const torch::Tensor& w13_scale,
    const torch::Tensor& topk_ids,
    torch::Tensor& intermediate,
    int num_experts,
    int inter_size,
    int hidden,
    int topk,
    int num_routes,
    float limit_bf16) {
  using Kernel = MoEW4A16GemvUpKernel<index_t>;
  const int blocks = inter_size / (Kernel::GATE_ROWS * SG_PER_WG);
  Kernel task(
      reinterpret_cast<const uint32_t*>(hidden_states.data_ptr()),
      reinterpret_cast<const uint8_t*>(w13.data_ptr()),
      reinterpret_cast<const uint8_t*>(w13_scale.data_ptr()),
      reinterpret_cast<const index_t*>(topk_ids.data_ptr()),
      reinterpret_cast<bf16_t*>(intermediate.data_ptr()),
      num_experts,
      inter_size,
      hidden,
      topk,
      num_routes,
      limit_bf16);
  sycl_kernel_submit(static_cast<int64_t>(num_routes) * blocks * WG_SIZE, WG_SIZE, queue, task);
}

template <typename index_t, typename weight_t, int TOPK, int LPR>
void launch_down(
    sycl::queue& queue,
    const torch::Tensor& intermediate,
    const torch::Tensor& w2,
    const torch::Tensor& w2_scale,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    torch::Tensor& output,
    int num_tokens,
    int num_experts,
    int hidden,
    float routed_scaling_factor) {
  using Kernel = MoEW4A16GemvDownKernel<index_t, weight_t, TOPK, LPR>;
  const int blocks = hidden / (Kernel::ROWS * SG_PER_WG);
  Kernel task(
      reinterpret_cast<const uint32_t*>(intermediate.data_ptr()),
      reinterpret_cast<const uint8_t*>(w2.data_ptr()),
      reinterpret_cast<const uint8_t*>(w2_scale.data_ptr()),
      reinterpret_cast<const index_t*>(topk_ids.data_ptr()),
      reinterpret_cast<const weight_t*>(topk_weights.data_ptr()),
      reinterpret_cast<bf16_t*>(output.data_ptr()),
      num_tokens,
      num_experts,
      hidden,
      routed_scaling_factor);
  sycl_kernel_submit(static_cast<int64_t>(num_tokens) * blocks * WG_SIZE, WG_SIZE, queue, task);
}

inline bool aligned16(const torch::Tensor& t) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

}  // namespace moe_w4a16_gemv

SGL_KERNEL_EXPORT void moe_w4a16_decode_gemv(
    torch::Tensor& output,               // [T, H] bf16
    const torch::Tensor& hidden_states,  // [T, H] bf16
    const torch::Tensor& w13,            // [E, 2I, H/2] int8/uint8
    const torch::Tensor& w13_scale,      // [E, 2I, H/32] uint8/float8_e8m0fnu
    const torch::Tensor& w2,             // [E, H, I/2] int8/uint8
    const torch::Tensor& w2_scale,       // [E, H, I/32] uint8/float8_e8m0fnu
    const torch::Tensor& topk_weights,   // [T, topk] float32/bf16
    const torch::Tensor& topk_ids,       // [T, topk] int32/int64
    torch::Tensor& intermediate,         // [T * topk, I] bf16 workspace
    double routed_scaling_factor,
    double swiglu_limit) {
  using namespace moe_w4a16_gemv;
  CHECK_INPUT(output);
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(w13);
  CHECK_INPUT(w13_scale);
  CHECK_INPUT(w2);
  CHECK_INPUT(w2_scale);
  CHECK_INPUT(topk_weights);
  CHECK_INPUT(topk_ids);
  CHECK_INPUT(intermediate);

  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D [T, H]");
  TORCH_CHECK(hidden_states.scalar_type() == at::ScalarType::BFloat16, "hidden_states must be bfloat16");
  TORCH_CHECK(output.sizes() == hidden_states.sizes(), "output must have the shape of hidden_states");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16, "output must be bfloat16");
  TORCH_CHECK(w13.dim() == 3 && w2.dim() == 3, "w13 and w2 must be 3D");
  TORCH_CHECK(
      w13.scalar_type() == at::ScalarType::Char || w13.scalar_type() == at::ScalarType::Byte,
      "w13 must be int8 or uint8 (packed E2M1)");
  TORCH_CHECK(
      w2.scalar_type() == at::ScalarType::Char || w2.scalar_type() == at::ScalarType::Byte,
      "w2 must be int8 or uint8 (packed E2M1)");
  TORCH_CHECK(
      w13_scale.scalar_type() == at::ScalarType::Byte || w13_scale.scalar_type() == at::ScalarType::Float8_e8m0fnu,
      "w13_scale must be uint8 or float8_e8m0fnu");
  TORCH_CHECK(
      w2_scale.scalar_type() == at::ScalarType::Byte || w2_scale.scalar_type() == at::ScalarType::Float8_e8m0fnu,
      "w2_scale must be uint8 or float8_e8m0fnu");

  const int64_t num_tokens = hidden_states.size(0);
  const int64_t hidden = hidden_states.size(1);
  const int64_t num_experts = w13.size(0);
  const int64_t inter_size = w13.size(1) / 2;
  const int64_t topk = topk_ids.dim() == 2 ? topk_ids.size(1) : 0;
  const int64_t num_routes = num_tokens * topk;

  TORCH_CHECK(w13.size(1) == 2 * inter_size && w13.size(2) == hidden / 2, "w13 must be [E, 2I, H/2]");
  TORCH_CHECK(
      w2.size(0) == num_experts && w2.size(1) == hidden && w2.size(2) == inter_size / 2, "w2 must be [E, H, I/2]");
  TORCH_CHECK(
      w13_scale.size(0) == num_experts && w13_scale.size(1) == 2 * inter_size && w13_scale.size(2) == hidden / GROUP,
      "w13_scale must be [E, 2I, H/32] (MXFP4 group size 32)");
  TORCH_CHECK(
      w2_scale.size(0) == num_experts && w2_scale.size(1) == hidden && w2_scale.size(2) == inter_size / GROUP,
      "w2_scale must be [E, H, I/32] (MXFP4 group size 32)");
  TORCH_CHECK(hidden % SG_K_VALUES == 0, "hidden size must be a multiple of ", SG_K_VALUES);
  TORCH_CHECK(hidden % (4 * SG_PER_WG) == 0, "hidden size must be a multiple of ", 4 * SG_PER_WG);
  TORCH_CHECK(inter_size == 256 || inter_size == 512, "intermediate size per partition must be 256 or 512");
  TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.size(0) == num_tokens, "topk_ids must be [T, topk]");
  TORCH_CHECK(topk_weights.sizes() == topk_ids.sizes(), "topk_weights must have the shape of topk_ids");
  TORCH_CHECK(topk == 6 || topk == 8, "topk must be 6 or 8");
  TORCH_CHECK(num_routes <= MAX_ROUTES, "T * topk must be at most ", MAX_ROUTES, " for the decode GEMV");
  TORCH_CHECK(
      topk_ids.scalar_type() == at::ScalarType::Int || topk_ids.scalar_type() == at::ScalarType::Long,
      "topk_ids must be int32 or int64");
  TORCH_CHECK(
      topk_weights.scalar_type() == at::ScalarType::Float || topk_weights.scalar_type() == at::ScalarType::BFloat16,
      "topk_weights must be float32 or bfloat16");
  TORCH_CHECK(
      intermediate.scalar_type() == at::ScalarType::BFloat16 && intermediate.dim() == 2 &&
          intermediate.size(0) == num_routes && intermediate.size(1) == inter_size,
      "intermediate must be bfloat16 [T * topk, I]");
  TORCH_CHECK(
      aligned16(hidden_states) && aligned16(w13) && aligned16(w2) && aligned16(intermediate),
      "hidden_states, w13, w2 and intermediate must be 16-byte aligned");
  TORCH_CHECK(swiglu_limit > 0.0, "swiglu_limit must be positive");
  const auto device = hidden_states.device();
  for (const torch::Tensor* t : std::initializer_list<const torch::Tensor*>{
           &output, &w13, &w13_scale, &w2, &w2_scale, &topk_weights, &topk_ids, &intermediate}) {
    TORCH_CHECK(t->device() == device, "all tensors must be on the same device");
  }

  if (num_tokens == 0) return;

  c10::DeviceGuard device_guard(hidden_states.device());
  auto queue = at::xpu::getCurrentXPUStream().queue();
  const float limit_bf16 = static_cast<float>(static_cast<bf16_t>(static_cast<float>(swiglu_limit)));
  const float rsf = static_cast<float>(routed_scaling_factor);

  const bool ids_i64 = topk_ids.scalar_type() == at::ScalarType::Long;
  const bool weights_f32 = topk_weights.scalar_type() == at::ScalarType::Float;

  auto run = [&](auto index_tag, auto weight_tag) {
    using index_t = decltype(index_tag);
    using weight_t = decltype(weight_tag);
    launch_up<index_t>(
        queue,
        hidden_states,
        w13,
        w13_scale,
        topk_ids,
        intermediate,
        static_cast<int>(num_experts),
        static_cast<int>(inter_size),
        static_cast<int>(hidden),
        static_cast<int>(topk),
        static_cast<int>(num_routes),
        limit_bf16);
    // One row of w2 spans I / 32 lanes: 8 at I = 256 (TP 8), 16 at I = 512.
    auto down = [&](auto topk_tag, auto lpr_tag) {
      constexpr int TOPK = decltype(topk_tag)::value;
      constexpr int LPR = decltype(lpr_tag)::value;
      launch_down<index_t, weight_t, TOPK, LPR>(
          queue,
          intermediate,
          w2,
          w2_scale,
          topk_ids,
          topk_weights,
          output,
          static_cast<int>(num_tokens),
          static_cast<int>(num_experts),
          static_cast<int>(hidden),
          rsf);
    };
    using I6 = std::integral_constant<int, 6>;
    using I8 = std::integral_constant<int, 8>;
    using I16 = std::integral_constant<int, 16>;
    if (topk == 6 && inter_size == 256) {
      down(I6{}, I8{});
    } else if (topk == 6) {
      down(I6{}, I16{});
    } else if (inter_size == 256) {
      down(I8{}, I8{});
    } else {
      down(I8{}, I16{});
    }
  };

  if (ids_i64) {
    if (weights_f32) {
      run(int64_t{}, float{});
    } else {
      run(int64_t{}, bf16_t{});
    }
  } else {
    if (weights_f32) {
      run(int32_t{}, float{});
    } else {
      run(int32_t{}, bf16_t{});
    }
  }
}
