#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <optional>
#include <tuple>

namespace ds_flash_mla_moe::moe {

std::tuple<at::Tensor, at::Tensor> grouped_topk_cuda_entry(
    const at::Tensor& x,
    const at::Tensor& gate_weight,
    int64_t topk,
    int64_t n_groups,
    int64_t topk_groups,
    const std::optional<at::Tensor>& score_bias,
    double route_scale);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
route_pack_cuda_entry(
    const at::Tensor& x,
    const at::Tensor& route_weights,
    const at::Tensor& expert_indices,
    const at::Tensor& expert_owner,
    int64_t world_size);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
single_device_route_pack_cuda_entry(
    const at::Tensor& x,
    const at::Tensor& route_weights,
    const at::Tensor& expert_indices,
    int64_t experts);

at::Tensor swiglu_experts_cuda_entry(
    const at::Tensor& activations,
    const at::Tensor& expert_offsets,
    const at::Tensor& expert_w1,
    const at::Tensor& expert_w2,
    const at::Tensor& expert_w3);

at::Tensor swiglu_experts_fused_cuda_entry(
    const at::Tensor& activations,
    const at::Tensor& expert_offsets,
    const at::Tensor& packed_weights,
    const at::Tensor& token_indices,
    const at::Tensor& expert_w1,
    const at::Tensor& expert_w2,
    const at::Tensor& expert_w3,
    int64_t token_count);

at::Tensor swiglu_experts_persistent_cuda_entry(
    const at::Tensor& activations,
    const at::Tensor& expert_offsets,
    const at::Tensor& packed_weights,
    const at::Tensor& token_indices,
    const at::Tensor& expert_w1,
    const at::Tensor& expert_w2,
    const at::Tensor& expert_w3,
    int64_t token_count);

at::Tensor route_combine_cuda_entry(
    const at::Tensor& contributions,
    const at::Tensor& route_weights,
    const at::Tensor& token_indices,
    int64_t token_count);

}  // namespace ds_flash_mla_moe::moe
