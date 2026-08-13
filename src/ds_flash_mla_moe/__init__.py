"""Executable specifications for DeepSeek-style attention and MoE operators."""

from .attention import (
    blockwise_attention,
    scaled_dot_product_attention_backward_reference,
    scaled_dot_product_attention_reference,
)
from .expert_ops import (
    cuda_expert_ops_available,
    expert_major_pack,
    swiglu_experts_expert_major,
)
from .expert_parallel import (
    ExpertComputeBackend,
    ExpertParallelTrace,
    deepseek_moe_expert_parallel_reference,
)
from .gemm import gemm_reference, tiled_gemm_reference
from .mla import (
    MLAConfig,
    MLALatentCache,
    MLAStaticCache,
    MLAWeights,
    allocate_mla_static_cache,
    append_mla_cache,
    build_mla_cache,
    mla_absorbed_attention_reference,
    mla_naive_attention_reference,
    write_mla_static_cache,
)
from .moe import (
    ExpertMajorLayout,
    PackedRoutes,
    RoutingResult,
    combine_packed_routes,
    deepseek_grouped_topk,
    deepseek_moe_packed_reference,
    deepseek_moe_reference,
    pack_routes_reference,
    swiglu_expert,
    swiglu_experts_expert_major_reference,
    swiglu_experts_padded_reference,
    to_expert_major_reference,
)
from .ops import (
    cuda_gemm_available,
    cuda_kernel_available,
    flash_attention_forward,
    native_extension_loaded,
    tiled_gemm,
)
from .route_ops import RoutePackResult, cuda_route_ops_available, route_combine, route_pack
from .router_ops import RouterBackend, cuda_router_available, grouped_topk
from .symmetric_memory import (
    SymmetricMoEBufferLayout,
    symmetric_moe_buffer_estimate,
    symmetric_moe_buffer_model_from_routes,
    symmetric_moe_buffer_offset,
)
from .version import __version__

__all__ = [
    "ExpertComputeBackend",
    "ExpertMajorLayout",
    "ExpertParallelTrace",
    "MLAConfig",
    "MLALatentCache",
    "MLAStaticCache",
    "MLAWeights",
    "PackedRoutes",
    "RoutePackResult",
    "RouterBackend",
    "RoutingResult",
    "SymmetricMoEBufferLayout",
    "__version__",
    "allocate_mla_static_cache",
    "append_mla_cache",
    "blockwise_attention",
    "build_mla_cache",
    "combine_packed_routes",
    "cuda_expert_ops_available",
    "cuda_gemm_available",
    "cuda_kernel_available",
    "cuda_route_ops_available",
    "cuda_router_available",
    "deepseek_grouped_topk",
    "deepseek_moe_expert_parallel_reference",
    "deepseek_moe_packed_reference",
    "deepseek_moe_reference",
    "expert_major_pack",
    "flash_attention_forward",
    "gemm_reference",
    "grouped_topk",
    "mla_absorbed_attention_reference",
    "mla_naive_attention_reference",
    "native_extension_loaded",
    "pack_routes_reference",
    "route_combine",
    "route_pack",
    "scaled_dot_product_attention_backward_reference",
    "scaled_dot_product_attention_reference",
    "swiglu_expert",
    "swiglu_experts_expert_major",
    "swiglu_experts_expert_major_reference",
    "swiglu_experts_padded_reference",
    "symmetric_moe_buffer_estimate",
    "symmetric_moe_buffer_model_from_routes",
    "symmetric_moe_buffer_offset",
    "tiled_gemm",
    "tiled_gemm_reference",
    "to_expert_major_reference",
    "write_mla_static_cache",
]
