#ifndef DS_FLASH_MLA_MOE_CSRC_ATTENTION_ATTENTION_COMMON_CUH_
#define DS_FLASH_MLA_MOE_CSRC_ATTENTION_ATTENTION_COMMON_CUH_

#include <ATen/ATen.h>

#include <cmath>
#include <cstdint>

namespace ds_flash_mla_moe::attention {

constexpr int kWarpSize = 32;
constexpr int kWarps = 4;
constexpr int kThreads = kWarpSize * kWarps;
constexpr int kQueryTile = 4;
constexpr int kKeyTile = 16;
constexpr int kMaxHeadDim = 128;
constexpr int kMaxValueDim = 128;

__device__ __forceinline__ bool key_is_visible(
    int64_t query_position,
    int64_t key_position,
    int64_t query_length,
    int64_t key_length,
    bool causal) {
  return !causal || key_position <= query_position + key_length - query_length;
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

inline void validate_formal_attention_forward_inputs(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    bool causal,
    double scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "q, k, and v must be CUDA tensors");
  TORCH_CHECK(!(q.requires_grad() || k.requires_grad() || v.requires_grad()),
              "formal FA1/FA2 forward kernels are forward-only and do not accept "
              "requires_grad tensors");
  TORCH_CHECK(q.device() == k.device() && k.device() == v.device(),
              "q, k, and v must be on the same CUDA device");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "q, k, and v must have shape [batch, heads, sequence, dimension]");
  TORCH_CHECK(q.scalar_type() == at::kHalf,
              "formal FA1/FA2 forward kernels support float16 only");
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
              "q, k, and v must have the same dtype");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "q, k, and v must be contiguous");
  TORCH_CHECK(q.size(0) == k.size(0) && k.size(0) == v.size(0) &&
                  q.size(1) == k.size(1) && k.size(1) == v.size(1),
              "q, k, and v must have identical batch and head dimensions");
  TORCH_CHECK(q.size(3) == k.size(3), "q and k must have the same head dimension");
  TORCH_CHECK(k.size(2) == v.size(2), "k and v must have the same sequence length");
  TORCH_CHECK(q.size(3) > 0, "attention head dimension must be positive");
  TORCH_CHECK(k.size(2) > 0, "key sequence length must be positive");
  TORCH_CHECK(q.size(3) <= kMaxHeadDim,
              "formal FA1/FA2 require head_dim <= ", kMaxHeadDim);
  TORCH_CHECK(v.size(3) <= kMaxValueDim,
              "formal FA1/FA2 require value_dim <= ", kMaxValueDim);
  TORCH_CHECK(!causal || q.size(2) <= k.size(2),
              "right-aligned causal attention requires query_length <= key_length");
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");
}

}  // namespace ds_flash_mla_moe::attention

#endif  // DS_FLASH_MLA_MOE_CSRC_ATTENTION_ATTENTION_COMMON_CUH_
