# ds-flashDMLA-moe

CUDA/C++ operators and minimal Python bindings for a FlashAttention-style **DMLA** (Multi-head Latent Attention) backend and a  **DeepSeek-V3–style MoE FFN megakernel** .

The project provides forward **and** backward paths, single-node multi-GPU execution with **data parallel (DP)** and  **expert parallel (EP)** , and numerically consistent I/O with the DeepSeek-V3 MoE interface.

---

## Key capabilities

* **Flash-DMLA attention (forward/backward)**
  * Online/streaming softmax; row-wise statistics (`m`, `l`) saved for backward.
  * Two execution modes:
    * **On-the-fly** : project `K,V → K_lat,V_lat` inside the kernel (`WK`,`WV` learnable).
    * **Cache-latent** : consume precomputed `K_lat,V_lat` from the KV-cache.
  * Drop-in interface compatible with standard MHA shapes.
* **MoE FFN**
  * **Unfused reference path** : Top-K gate → per-expert packing (bucket + exclusive scan) → expert MLP (batched) → combine & shared-expert add.
  * **Fused megakernel** : gate → pack/dispatch → expert MLP → reduce/restore → shared add in one or few kernels (forward/backward).
  * Gate weight renormalization under capacity; overflow tokens are masked.
* **Distributed single-node DP×EP**
  * NCCL collectives for EP: `all_to_allv`, `all_gather`, `reduce_scatter`, `all_reduce`.
  * DP for parameter/gradient synchronization; no tensor parallel required.
* **Minimal PyTorch integration**
  * Custom `autograd.Function` wrappers; ATen/pybind11 bindings.

---

## Repository layout

<pre class="overflow-visible!" data-start="1505" data-end="2152"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>ds-flashdMLA-moe/
├─ README.md
├─ pyproject.toml               </span><span># or setup.py</span><span>
├─ csrc/
│  ├─ torch_bindings.cpp        </span><span># pybind11/ATen entry points</span><span>
│  ├─ dist/
│  │  ├─ nccl_comm.h
│  │  └─ nccl_comm.cpp          </span><span># EP group + A2A/Gather/RS/AR helpers</span><span>
│  ├─ attention/
│  │  ├─ vanalla_attn.cu
│  │  ├─ fa_fwd.cu
│  │  ├─ fa2_fwd.cu
│  │  ├─ fa2_bwd.cu
│  │  ├─ fa2_bwd_mpi.cu
│  │  ├─ fa2_dmla_fwd.cu
│  │  └─ fa2_dmla_bwd.cu
│  └─ moe/
│     ├─ router_topk.cu
│     ├─ pack_dispatch.cu
│     ├─ expert_mlp.cu
│     ├─ combine_scatter.cu
│     ├─ moe_megakernel_fwd.cu
│     └─ moe_megakernel_bwd.cu
├─ python/
│  ├─ __init__.py
│  └─ ops.py                    </span><span># minimal autograd.Function wrappers</span><span>
└─ tests/
   ├─ test_attn.py
   └─ test_moe.py
</span></span></code></div></div></pre>
