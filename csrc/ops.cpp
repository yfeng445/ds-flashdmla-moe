#include <Python.h>
#include <torch/library.h>

extern "C" PyObject* PyInit__C() {
  static PyModuleDef module = {
      PyModuleDef_HEAD_INIT,
      "_C",
      nullptr,
      -1,
      nullptr,
  };
  return PyModule_Create(&module);
}

TORCH_LIBRARY(ds_flash_mla_moe, m) {
  m.def(
      "attention_forward(Tensor q, Tensor k, Tensor v, bool causal, float scale) -> Tensor");
  m.def(
      "attention_backward(Tensor grad_output, Tensor q, Tensor k, Tensor v, bool causal, float scale) -> (Tensor, Tensor, Tensor)");
  m.def(
      "route_pack(Tensor x, Tensor route_weights, Tensor expert_indices, Tensor expert_owner, int world_size) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "route_combine(Tensor contributions, Tensor route_weights, Tensor token_indices, int token_count) -> Tensor");
  m.def(
      "tiled_gemm(Tensor a, Tensor b, Tensor? c, float alpha, float beta) -> Tensor");
  m.def(
      "swiglu_experts(Tensor activations, Tensor expert_offsets, Tensor expert_w1, Tensor expert_w2, Tensor expert_w3) -> Tensor");
  m.def(
      "expert_major_pack(Tensor activations, Tensor expert_indices, Tensor local_expert_ids) -> (Tensor, Tensor, Tensor)");
  m.def(
      "grouped_topk(Tensor x, Tensor gate_weight, int topk, int n_groups, int topk_groups, Tensor? score_bias, float route_scale) -> (Tensor, Tensor)");
  m.def(
      "mla_absorbed_attention(Tensor q_nope, Tensor q_pe, Tensor kv, Tensor pe, Tensor key_up, Tensor value_up, Tensor query_positions, Tensor key_positions, bool causal, float scale) -> Tensor");
}
