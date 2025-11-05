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
│  │  ├─ flash_dmla_fwd.cu
│  │  └─ flash_dmla_bwd.cu
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

---

## Build

**Requirements**

* CUDA 12.x, cuDNN as provided by your CUDA toolkit
* PyTorch ≥ 2.3 with matching CUDA
* C++17 toolchain, NCCL available on the system

**Compile & install (editable)**

<pre class="overflow-visible!" data-start="2356" data-end="2476"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python -m pip install -e .
</span><span># Optionally set architectures, e.g.:</span><span>
</span><span># export TORCH_CUDA_ARCH_LIST="80;86;89;90"</span><span>
</span></span></code></div></div></pre>

---

## Usage

### Flash-DMLA (forward/backward)

<pre class="overflow-visible!" data-start="2528" data-end="3260"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-python"><span><span>import</span><span> torch
</span><span>import</span><span> ds_flashdmla_moe </span><span>as</span><span> dsm

B, H, Tq, Tk, Dq, Dk, Dv, D_lat = </span><span>2</span><span>, </span><span>16</span><span>, </span><span>512</span><span>, </span><span>512</span><span>, </span><span>128</span><span>, </span><span>128</span><span>, </span><span>128</span><span>, </span><span>64</span><span>
Q = torch.randn(B,H,Tq,Dq, device=</span><span>"cuda"</span><span>, dtype=torch.float16, requires_grad=</span><span>True</span><span>)
K = torch.randn(B,H,Tk,Dk, device=</span><span>"cuda"</span><span>, dtype=torch.float16, requires_grad=</span><span>True</span><span>)
V = torch.randn(B,H,Tk,Dv, device=</span><span>"cuda"</span><span>, dtype=torch.float16, requires_grad=</span><span>True</span><span>)
WK = torch.randn(Dk, D_lat, device=</span><span>"cuda"</span><span>, dtype=torch.float16, requires_grad=</span><span>True</span><span>)
WV = torch.randn(Dv, D_lat, device=</span><span>"cuda"</span><span>, dtype=torch.float16, requires_grad=</span><span>True</span><span>)

</span><span># On-the-fly projection</span><span>
O = dsm.flash_dmla_fwd(Q, K, V, WK, WV, d_lat=D_lat, causal=</span><span>True</span><span>, cache_is_latent=</span><span>False</span><span>)
loss = O.square().mean()
loss.backward()  </span><span># produces dQ, dK, dV, dWK, dWV</span><span>
</span></span></code></div></div></pre>

### MoE FFN (unfused reference or fused megakernel)

<pre class="overflow-visible!" data-start="3315" data-end="3864"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-python"><span><span>import</span><span> torch
</span><span>import</span><span> ds_flashdmla_moe </span><span>as</span><span> dsm

B, T, D_model, D_hidden = </span><span>8</span><span>, </span><span>1024</span><span>, </span><span>2048</span><span>, </span><span>8192</span><span>
topk, capacity = </span><span>8</span><span>, </span><span>1.25</span><span>

X = torch.randn(B,T,D_model, device=</span><span>"cuda"</span><span>, dtype=torch.float16, requires_grad=</span><span>True</span><span>)

</span><span># Unfused reference (acts as numerical baseline)</span><span>
Y_ref = dsm.moe_forward_unfused(X, topk=topk, capacity=capacity, ep_group_size=</span><span>8</span><span>)

</span><span># Fused megakernel (I/O equivalent to unfused)</span><span>
Y = dsm.moe_megakernel_fwd(X, topk=topk, capacity=capacity, ep_group_size=</span><span>8</span><span>)

</span><span># Training step</span><span>
loss = (Y - Y_ref.detach()).square().mean()
loss.backward()
</span></span></code></div></div></pre>

**Distributed launch (single node)**

<pre class="overflow-visible!" data-start="3904" data-end="4058"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun --nproc_per_node=8 -m your_training_entry.py \
  --ep_size 8 --dp_size 1 --topk 8 --capacity 1.25
</span></span></code></div></div></pre>

EP/DP groups are created internally; environment variables for NCCL (e.g., `NCCL_DEBUG=INFO`) can be added as needed.

---

## Configuration

* `d_lat` — latent dimension for DMLA projection.
* `cache_is_latent` — `True` if the KV-cache stores `K_lat,V_lat`; otherwise the kernel computes them via `WK,WV`.
* `topk` — number of active experts per token in the routed branch.
* `capacity` — capacity factor; tokens exceeding per-expert capacity are masked.
* `ep_group_size`, `dp_group_size` — process group sizes; EP handles A2A routing, DP reduces gradients.

---

## Numerical contracts

* **Attention** : forward uses online softmax; backward reconstructs stable gradients from saved row statistics.
* **DMLA modes** :
* *On-the-fly* : returns `dWK,dWV` and propagates `dK,dV` via projection.
* *Cache-latent* : no projection gradients; gradients are applied to the latent producers.
* **MoE** :
* Gate logits → softmax → Top-K; weights are renormalized over the selected set.
* Capacity/overflow masking yields zero contribution and zero gradient for dropped items.
* Expert outputs are combined with FP32 accumulation; shared-expert output is added per token.
* **Distributed EP** :
* Two A2A phases (dispatch/restore) preserve token order via saved indices and per-expert offsets (exclusive-scan).
* DP uses `all_reduce` for parameter/gradient synchronization.

---

## Testing

<pre class="overflow-visible!" data-start="5455" data-end="5673"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span># attention numerical checks (forward/backward, multiple dtypes/shapes)</span><span>
pytest -q tests/test_attn.py

</span><span># MoE numerical checks (unfused vs fused I/O equivalence; DP×EP single node)</span><span>
pytest -q tests/test_moe.py
</span></span></code></div></div></pre>

Tests verify:

* Forward closeness (relative/absolute tolerances by dtype)
* Backward consistency (gradients on Q/K/V and projections; MoE gate/MLP weights)
* EP dispatch/restore shape/order invariants

---

## Design notes

* **Online softmax** : forward stores per-row `(m,l)` statistics; backward uses the stable softmax gradient form without materializing `P`.
* **Per-expert packing** : warp/block-level **exclusive scan** generates contiguous write offsets; memory layout is expert-major for efficient batched GEMM.
* **Megakernel outline** : gate → pack → expert MLP → combine in a persistent, staged kernel; FP32 accumulation at merge points.
* **Autograd** : saved tensors are minimized; recomputation is favored over large intermediates.

---

## License

MIT (or BSD-3-Clause). Please choose one and include the header banner in source files.
