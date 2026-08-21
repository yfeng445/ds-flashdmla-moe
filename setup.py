"""Optional native build hook; the default wheel remains pure Python."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup


def native_build_configuration() -> dict[str, object]:
    if os.environ.get("DS_FLASH_BUILD_CUDA") != "1":
        return {}

    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    cxx_args = ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-std=c++17"]

    extension = CUDAExtension(
        name="ds_flash_mla_moe._C",
        sources=[
            "csrc/ops.cpp",
            "csrc/attention/attention_backward_cuda.cu",
            "csrc/attention/fa1_forward_cuda.cu",
            "csrc/attention/fa2_forward_cuda.cu",
            "csrc/attention/attention_forward_cuda.cu",
            "csrc/gemm/tiled_gemm_cuda.cu",
            "csrc/moe/route_ops_cuda.cu",
            "csrc/moe/expert_major_pack_cuda.cu",
            "csrc/moe/grouped_topk_cuda.cu",
            "csrc/moe/swiglu_experts_cuda.cu",
            "csrc/moe/deepseek_moe_forward_cuda.cu",
            "csrc/moe/deepseek_moe_forward_fused_cuda.cu",
            "csrc/mla/mla_absorbed_attention_cuda.cu",
            "csrc/mla/mla_paged_attention_cuda.cu",
            "csrc/mla/mla_projection_cuda.cu",
        ],
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": ["-O3", "-std=c++17", "-lineinfo"],
        },
    )
    return {
        "ext_modules": [extension],
        "cmdclass": {"build_ext": BuildExtension.with_options(use_ninja=True)},
    }


def assert_native_sources_present() -> None:
    if os.environ.get("DS_FLASH_BUILD_CUDA") != "1":
        return
    required = (
        Path("csrc/ops.cpp"),
        Path("csrc/attention/attention_backward_cuda.cu"),
        Path("csrc/attention/fa1_forward_cuda.cu"),
        Path("csrc/attention/fa2_forward_cuda.cu"),
        Path("csrc/attention/attention_forward_cuda.cu"),
        Path("csrc/gemm/tiled_gemm_cuda.cu"),
        Path("csrc/moe/route_ops_cuda.cu"),
        Path("csrc/moe/expert_major_pack_cuda.cu"),
        Path("csrc/moe/grouped_topk_cuda.cu"),
        Path("csrc/moe/swiglu_experts_cuda.cu"),
        Path("csrc/moe/deepseek_moe_forward_cuda.cu"),
        Path("csrc/moe/deepseek_moe_forward_fused_cuda.cu"),
        Path("csrc/mla/mla_absorbed_attention_cuda.cu"),
        Path("csrc/mla/mla_paged_attention_cuda.cu"),
        Path("csrc/mla/mla_projection_cuda.cu"),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"native build sources are missing: {', '.join(missing)}")


assert_native_sources_present()
setup(**native_build_configuration())
