#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <torch/library.h>

#include "moe_cuda_ops.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <tuple>

namespace {

void check_forward_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must use float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      !tensor.requires_grad(),
      "the DeepSeek MoE forward operator is forward-only and does not accept ",
      "requires_grad tensors");
}

}  // namespace

namespace ds_flash_mla_moe::moe {

at::Tensor deepseek_moe_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& gate_weight,
    const at::Tensor& expert_w1,
    const at::Tensor& expert_w2,
    const at::Tensor& expert_w3,
    int64_t topk,
    int64_t n_groups,
    int64_t topk_groups,
    const std::optional<at::Tensor>& score_bias,
    double route_scale) {
  TORCH_CHECK(
      !at::globalContext().deterministicAlgorithms(),
      "the CUDA DeepSeek MoE forward uses atomic routing; disable deterministic algorithms");
  check_forward_tensor(x, "x");
  check_forward_tensor(gate_weight, "gate_weight");
  check_forward_tensor(expert_w1, "expert_w1");
  check_forward_tensor(expert_w2, "expert_w2");
  check_forward_tensor(expert_w3, "expert_w3");
  TORCH_CHECK(x.dim() == 2, "x must have shape [tokens, model_dim]");
  TORCH_CHECK(gate_weight.dim() == 2,
              "gate_weight must have shape [experts, model_dim]");
  TORCH_CHECK(
      expert_w1.dim() == 3 && expert_w2.dim() == 3 && expert_w3.dim() == 3,
      "routed expert weights must be rank-3 tensors");
  TORCH_CHECK(
      x.device() == gate_weight.device() && x.device() == expert_w1.device() &&
          x.device() == expert_w2.device() && x.device() == expert_w3.device(),
      "all DeepSeek MoE tensors must be on the same CUDA device");

  const int64_t experts = gate_weight.size(0);
  const int64_t model_dim = gate_weight.size(1);
  const int64_t hidden_dim = expert_w1.size(1);
  TORCH_CHECK(experts > 0, "number of experts must be positive");
  TORCH_CHECK(model_dim > 0, "model_dim must be positive");
  TORCH_CHECK(hidden_dim > 0, "hidden_dim must be positive");
  TORCH_CHECK(x.size(1) == model_dim,
              "x model dimension does not match gate_weight");
  TORCH_CHECK(
      expert_w1.size(0) == experts && expert_w1.size(2) == model_dim,
      "expert_w1 must have shape [experts, hidden_dim, model_dim]");
  TORCH_CHECK(
      expert_w2.size(0) == experts && expert_w2.size(1) == model_dim &&
          expert_w2.size(2) == hidden_dim,
      "expert_w2 must have shape [experts, model_dim, hidden_dim]");
  TORCH_CHECK(expert_w3.sizes() == expert_w1.sizes(),
              "expert_w3 shape must match expert_w1");

  if (score_bias.has_value()) {
    check_forward_tensor(*score_bias, "score_bias");
    TORCH_CHECK(score_bias->device() == x.device(),
                "score_bias must be on the same CUDA device as x");
    TORCH_CHECK(
        score_bias->dim() == 1 && score_bias->numel() == experts,
        "score_bias must have shape [experts]");
  }

  TORCH_CHECK(
      n_groups > 0 && experts % n_groups == 0,
      "number of experts must be divisible by n_groups");
  TORCH_CHECK(topk_groups >= 1 && topk_groups <= n_groups,
              "topk_groups must be in [1, n_groups]");
  TORCH_CHECK(topk >= 1 && topk <= experts,
              "topk must be in [1, experts]");
  TORCH_CHECK(
      topk <= topk_groups * (experts / n_groups),
      "topk exceeds the experts retained by group selection");
  TORCH_CHECK(std::isfinite(route_scale), "route_scale must be finite");

  auto expert_owner = at::zeros(
      {experts},
      x.options().dtype(at::kLong));
  auto routing = grouped_topk_cuda_entry(
      x,
      gate_weight,
      topk,
      n_groups,
      topk_groups,
      score_bias,
      route_scale);
  auto packed_routes = route_pack_cuda_entry(
      x,
      std::get<0>(routing),
      std::get<1>(routing),
      expert_owner,
      1);
  const auto& counts_per_expert = std::get<4>(packed_routes);
  auto expert_offsets = at::cat(
      {at::zeros({1}, counts_per_expert.options()),
       at::cumsum(counts_per_expert, 0)},
      0);
  auto contributions = swiglu_experts_cuda_entry(
      std::get<0>(packed_routes),
      expert_offsets,
      expert_w1,
      expert_w2,
      expert_w3);
  auto token_indices = at::floor_divide(std::get<2>(packed_routes), topk);
  return route_combine_cuda_entry(
      contributions,
      std::get<1>(packed_routes),
      token_indices,
      x.size(0));
}

}  // namespace ds_flash_mla_moe::moe

TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl(
      "deepseek_moe_forward",
      TORCH_FN(ds_flash_mla_moe::moe::deepseek_moe_forward_cuda));
}
