#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "moe_cuda_ops.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <optional>
#include <tuple>

namespace {

constexpr int kThreads = 256;

__device__ float group_score(
    const float* __restrict__ scores,
    const float* __restrict__ score_bias,
    int64_t group,
    int64_t experts_per_group) {
  const int64_t start = group * experts_per_group;
  if (score_bias == nullptr) {
    float maximum = -CUDART_INF_F;
    for (int64_t offset = 0; offset < experts_per_group; ++offset) {
      maximum = fmaxf(maximum, scores[start + offset]);
    }
    return maximum;
  }

  float first = -CUDART_INF_F;
  float second = -CUDART_INF_F;
  for (int64_t offset = 0; offset < experts_per_group; ++offset) {
    const int64_t expert = start + offset;
    const float value = scores[expert] + score_bias[expert];
    if (value > first) {
      second = first;
      first = value;
    } else if (value > second) {
      second = value;
    }
  }
  return experts_per_group == 1 ? first : first + second;
}

__device__ bool group_is_selected(
    const float* __restrict__ scores,
    const float* __restrict__ score_bias,
    int64_t candidate_group,
    int64_t n_groups,
    int64_t topk_groups,
    int64_t experts_per_group) {
  if (topk_groups == n_groups) {
    return true;
  }
  const float candidate =
      group_score(scores, score_bias, candidate_group, experts_per_group);
  int64_t rank = 0;
  for (int64_t group = 0; group < n_groups; ++group) {
    if (group == candidate_group) {
      continue;
    }
    const float other = group_score(scores, score_bias, group, experts_per_group);
    if (other > candidate || (other == candidate && group < candidate_group)) {
      ++rank;
    }
  }
  return rank < topk_groups;
}

__global__ void grouped_topk_select_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ score_bias,
    float* __restrict__ route_weights,
    int64_t* __restrict__ expert_indices,
    int64_t tokens,
    int64_t experts,
    int64_t topk,
    int64_t n_groups,
    int64_t topk_groups,
    float route_scale) {
  for (int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       token < tokens;
       token += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float* token_scores = scores + token * experts;
    float selected_sum = 0.0F;
    for (int64_t slot = 0; slot < topk; ++slot) {
      float best_score = -CUDART_INF_F;
      int64_t best_expert = -1;
      const int64_t experts_per_group = experts / n_groups;
      for (int64_t group = 0; group < n_groups; ++group) {
        if (!group_is_selected(
                token_scores,
                score_bias,
                group,
                n_groups,
                topk_groups,
                experts / n_groups)) {
          continue;
        }
        const int64_t group_start = group * experts_per_group;
        for (int64_t offset = 0; offset < experts_per_group; ++offset) {
          const int64_t expert = group_start + offset;
          bool already_selected = false;
          for (int64_t previous = 0; previous < slot; ++previous) {
            already_selected |= expert_indices[token * topk + previous] == expert;
          }
          if (already_selected) {
            continue;
          }
          const float selection_score =
              token_scores[expert] + (score_bias == nullptr ? 0.0F : score_bias[expert]);
          if (selection_score > best_score ||
              (selection_score == best_score && expert < best_expert)) {
            best_score = selection_score;
            best_expert = expert;
          }
        }
      }
      assert(best_expert >= 0 && best_expert < experts);
      expert_indices[token * topk + slot] = best_expert;
      const float unbiased_score = token_scores[best_expert];
      route_weights[token * topk + slot] = unbiased_score;
      selected_sum += unbiased_score;
    }
    const float denominator = fmaxf(selected_sum, FLT_MIN);
    for (int64_t slot = 0; slot < topk; ++slot) {
      route_weights[token * topk + slot] =
          route_weights[token * topk + slot] / denominator * route_scale;
    }
  }
}

void check_cuda_float_matrix(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must use float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2, name, " must be a rank-2 matrix");
}

std::tuple<at::Tensor, at::Tensor> grouped_topk_cuda(
    const at::Tensor& x,
    const at::Tensor& gate_weight,
    int64_t topk,
    int64_t n_groups,
    int64_t topk_groups,
    const std::optional<at::Tensor>& score_bias,
    double route_scale) {
  check_cuda_float_matrix(x, "x");
  check_cuda_float_matrix(gate_weight, "gate_weight");
  TORCH_CHECK(x.device() == gate_weight.device(),
              "x and gate_weight must be on the same CUDA device");
  TORCH_CHECK(x.size(1) == gate_weight.size(1),
              "x model dimension does not match gate_weight");
  const int64_t tokens = x.size(0);
  const int64_t experts = gate_weight.size(0);
  TORCH_CHECK(experts > 0, "number of experts must be positive");
  TORCH_CHECK(n_groups > 0 && experts % n_groups == 0,
              "experts must be divisible by n_groups");
  TORCH_CHECK(topk_groups >= 1 && topk_groups <= n_groups,
              "topk_groups must be in [1, n_groups]");
  TORCH_CHECK(topk >= 1 && topk <= experts,
              "topk must be in [1, experts]");
  TORCH_CHECK(topk <= topk_groups * (experts / n_groups),
              "topk exceeds the experts retained by group selection");
  TORCH_CHECK(std::isfinite(route_scale), "route_scale must be finite");
  if (score_bias.has_value()) {
    TORCH_CHECK(score_bias->is_cuda(), "score_bias must be a CUDA tensor");
    TORCH_CHECK(score_bias->scalar_type() == at::kFloat,
                "score_bias must use float32");
    TORCH_CHECK(score_bias->is_contiguous(), "score_bias must be contiguous");
    TORCH_CHECK(score_bias->device() == x.device(),
                "score_bias must be on the same CUDA device as x");
    TORCH_CHECK(score_bias->dim() == 1 && score_bias->numel() == experts,
                "score_bias must have shape [experts]");
  }
  TORCH_CHECK(tokens == 0 || experts <= INT64_MAX / tokens,
              "tokens * experts overflows int64");
  TORCH_CHECK(tokens == 0 || topk <= INT64_MAX / tokens,
              "tokens * topk overflows int64");

  const c10::cuda::CUDAGuard device_guard(x.device());
  auto scores = at::sigmoid(at::mm(x, gate_weight.transpose(0, 1)));
  auto route_weights = at::empty({tokens, topk}, x.options());
  auto expert_indices =
      at::empty({tokens, topk}, x.options().dtype(at::kLong));
  if (tokens == 0) {
    return {route_weights, expert_indices};
  }

  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(x.get_device());
  const int64_t blocks_unbounded = (tokens + kThreads - 1) / kThreads;
  const int64_t blocks = blocks_unbounded < properties->maxGridSize[0]
      ? blocks_unbounded
      : properties->maxGridSize[0];
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  const float* bias_pointer =
      score_bias.has_value() ? score_bias->const_data_ptr<float>() : nullptr;
  grouped_topk_select_kernel<<<
      static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
      scores.const_data_ptr<float>(),
      bias_pointer,
      route_weights.mutable_data_ptr<float>(),
      expert_indices.mutable_data_ptr<int64_t>(),
      tokens,
      experts,
      topk,
      n_groups,
      topk_groups,
      static_cast<float>(route_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {route_weights, expert_indices};
}

}  // namespace

namespace ds_flash_mla_moe::moe {

std::tuple<at::Tensor, at::Tensor> grouped_topk_cuda_entry(
    const at::Tensor& x,
    const at::Tensor& gate_weight,
    int64_t topk,
    int64_t n_groups,
    int64_t topk_groups,
    const std::optional<at::Tensor>& score_bias,
    double route_scale) {
  return grouped_topk_cuda(
      x,
      gate_weight,
      topk,
      n_groups,
      topk_groups,
      score_bias,
      route_scale);
}

}  // namespace ds_flash_mla_moe::moe

TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl(
      "grouped_topk",
      TORCH_FN(ds_flash_mla_moe::moe::grouped_topk_cuda_entry));
}
