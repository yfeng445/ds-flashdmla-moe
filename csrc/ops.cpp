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
      "attention_fa1_forward(Tensor q, Tensor k, Tensor v, bool causal, float scale) -> Tensor");
  m.def(
      "attention_fa2_forward(Tensor q, Tensor k, Tensor v, bool causal, float scale) -> Tensor");
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
  m.def(
      "mla_query_projection(Tensor x, Tensor wq, Tensor positions, int n_heads, int qk_nope_head_dim, int qk_rope_head_dim, float rope_theta) -> (Tensor, Tensor)");
  m.def(
      "mla_query_lora_projection(Tensor x, Tensor wq_a, Tensor q_norm_weight, Tensor wq_b, Tensor positions, int n_heads, int qk_nope_head_dim, int qk_rope_head_dim, float rope_theta, float rms_norm_eps) -> (Tensor, Tensor)");
  m.def(
      "mla_cache_projection(Tensor x, Tensor wkv_a, Tensor kv_norm_weight, Tensor positions, int kv_lora_rank, float rope_theta, float rms_norm_eps) -> (Tensor, Tensor)");
  m.def(
      "mla_cache_projection_write(Tensor x, Tensor wkv_a, Tensor kv_norm_weight, Tensor positions, Tensor(a!) kv_storage, Tensor(b!) pe_storage, Tensor(c!) position_storage, int start, float rope_theta, float rms_norm_eps) -> ()");
  m.def(
      "mla_cache_projection_write_slots(Tensor x, Tensor wkv_a, Tensor kv_norm_weight, Tensor positions, Tensor slot_mapping, Tensor(a!) kv_storage, Tensor(b!) pe_storage, Tensor(c!) position_storage, bool metadata_validated, float rope_theta, float rms_norm_eps) -> ()");
  m.def(
      "mla_paged_absorbed_attention(Tensor q_nope, Tensor q_pe, Tensor kv_storage, Tensor pe_storage, Tensor position_storage, Tensor block_table, Tensor sequence_lengths, Tensor key_up, Tensor value_up, Tensor query_positions, bool metadata_validated, bool causal, float scale) -> Tensor");
  m.def("mla_output_projection(Tensor heads, Tensor wo) -> Tensor");
}
