// vanilla_attention.cu
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>

static __device__ __forceinline__ float warp_reduce_max(float v){
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}
static __device__ __forceinline__ float warp_reduce_sum(float v){
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__global__ void vanilla_attention_row_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int Nq, int Nk, int d, int dv, float scale)
{
    const int i = blockIdx.x;
    if (i >= Nq) return;

    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int nwarps  = (blockDim.x + 31) >> 5;

    __shared__ float smem_red[32];

    const float* Qi = Q + (size_t)i * d;

    float local_max = -CUDART_INF_F;
    for (int j = threadIdx.x; j < Nk; j += blockDim.x){
        const float* Kj = K + (size_t)j * d;
        float dot = 0.f;
        #pragma unroll 1
        for (int k = 0; k < d; ++k) dot += Qi[k] * Kj[k];
        local_max = fmaxf(local_max, dot * scale);
    }
    float warp_maxv = warp_reduce_max(local_max);
    if (lane == 0) smem_red[warp_id] = warp_maxv;
    __syncthreads();
    if (warp_id == 0){
        float v = (lane < nwarps) ? smem_red[lane] : -CUDART_INF_F;
        v = warp_reduce_max(v);
        if (lane == 0) smem_red[0] = v;
    }
    __syncthreads();
    float row_max = smem_red[0];

    float local_sum = 0.f;
    for (int j = threadIdx.x; j < Nk; j += blockDim.x){
        const float* Kj = K + (size_t)j * d;
        float dot = 0.f;
        #pragma unroll 1
        for (int k = 0; k < d; ++k) dot += Qi[k] * Kj[k];
        local_sum += __expf(dot * scale - row_max);
    }
    float warp_sumv = warp_reduce_sum(local_sum);
    if (lane == 0) smem_red[warp_id] = warp_sumv;
    __syncthreads();
    if (warp_id == 0){
        float v = (lane < nwarps) ? smem_red[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem_red[0] = v;
    }
    __syncthreads();
    float row_sum = smem_red[0];

    for (int v = threadIdx.x; v < dv; v += blockDim.x){
        float acc = 0.f;
        for (int j = 0; j < Nk; ++j){
            const float* Kj = K + (size_t)j * d;
            float dot = 0.f;
            #pragma unroll 1
            for (int k = 0; k < d; ++k) dot += Qi[k] * Kj[k];
            float p = __expf(dot * scale - row_max) / row_sum;
            acc += p * V[(size_t)j * dv + v];
        }
        O[(size_t)i * dv + v] = acc;
    }
}

extern "C" void vanilla_attn_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int Nq, int Nk, int d, int dv,
    float scale,
    int /*warps_per_block*/,
    cudaStream_t stream)
{
    if (Nq <= 0 || Nk <= 0 || d <= 0 || dv <= 0) return;
    const int threads = 256;
    dim3 grid(Nq);
    vanilla_attention_row_kernel<<<grid, threads, 0, stream>>>(Q, K, V, O, Nq, Nk, d, dv, scale);
}
