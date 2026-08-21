"""Forward-only logical tensor-parallel SwiGLU reference."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class TensorParallelReport:
    """Single-device evidence boundary for one logical TP execution."""

    tp_size: int
    hidden_size: int
    shard_hidden_size: int
    accumulation_dtype: str
    simulated: bool = field(default=True, init=False)
    remote_visibility_verified: bool = field(default=False, init=False)
    transport_performed: bool = field(default=False, init=False)
    multi_gpu_verified: bool = field(default=False, init=False)

    def to_dict(self) -> dict[str, bool | int | str]:
        return {
            "simulated": self.simulated,
            "remote_visibility_verified": self.remote_visibility_verified,
            "transport_performed": self.transport_performed,
            "multi_gpu_verified": self.multi_gpu_verified,
            "tp_size": self.tp_size,
            "hidden_size": self.hidden_size,
            "shard_hidden_size": self.shard_hidden_size,
            "accumulation_dtype": self.accumulation_dtype,
        }


def _validate_swiglu_inputs(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
    tp_size: int,
) -> tuple[int, torch.dtype]:
    if isinstance(tp_size, bool) or tp_size not in (1, 2, 4):
        raise ValueError("tp_size must be 1, 2, or 4")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if w1.ndim != 2 or w2.ndim != 2 or w3.ndim != 2:
        raise ValueError("W1, W2, and W3 must be rank-2 tensors")
    hidden, model_dim = w1.shape
    if model_dim <= 0:
        raise ValueError("model dimension must be positive")
    if x.shape[-1] != model_dim:
        raise ValueError("x model dimension must match W1")
    if w3.shape != (hidden, model_dim):
        raise ValueError("W1 and W3 must have the same [hidden, model_dim] shape")
    if w2.shape != (model_dim, hidden):
        raise ValueError("W2 must have shape [model_dim, hidden]")
    if hidden <= 0 or hidden % tp_size:
        raise ValueError("hidden size must be positive and divisible by tp_size")
    tensors = (x, w1, w2, w3)
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise TypeError("all TP tensors must use floating-point dtypes")
    if any(tensor.dtype != x.dtype for tensor in tensors):
        raise ValueError("all TP tensors must use the same dtype")
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all TP tensors must use the same device")
    if any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("logical tensor-parallel SwiGLU is forward-only")
    accumulation_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return hidden, accumulation_dtype


def tensor_parallel_swiglu_forward(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
    *,
    tp_size: int,
    return_report: bool = False,
) -> Tensor | tuple[Tensor, TensorParallelReport]:
    """Evaluate column/row-sharded SwiGLU without performing communication.

    W1/W3 are sharded across hidden rows and W2 across matching hidden
    columns.  Partial outputs accumulate in FP64 for FP64 inputs and FP32 for
    every other supported floating dtype before one final cast.
    """

    hidden, accumulation_dtype = _validate_swiglu_inputs(x, w1, w2, w3, tp_size)
    shard_hidden = hidden // tp_size
    with torch.no_grad():
        x_acc = x.to(accumulation_dtype)
        partials: list[Tensor] = []
        for shard in range(tp_size):
            start = shard * shard_hidden
            end = start + shard_hidden
            gate = F.linear(x_acc, w1[start:end].to(accumulation_dtype))
            up = F.linear(x_acc, w3[start:end].to(accumulation_dtype))
            hidden_shard = F.silu(gate) * up
            partials.append(F.linear(hidden_shard, w2[:, start:end].to(accumulation_dtype)))
        output_acc = partials[0]
        for partial in partials[1:]:
            output_acc = output_acc + partial
        output = output_acc.to(x.dtype).contiguous()

    if not return_report:
        return output
    report = TensorParallelReport(
        tp_size=tp_size,
        hidden_size=hidden,
        shard_hidden_size=shard_hidden,
        accumulation_dtype=str(accumulation_dtype).removeprefix("torch."),
    )
    return output, report
