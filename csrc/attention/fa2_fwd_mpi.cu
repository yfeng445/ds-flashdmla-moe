#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>

extern "C" void fa2_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int Nq, int Nk, int d, int dv,
    float scale,
    int warps_per_block,
    cudaStream_t stream);

#define fa2_fwd_kernel fa2_forward

static int bind_cuda_device_to_local_rank(MPI_Comm comm) {
    MPI_Comm local_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    int dev = (ndev > 0) ? (local_rank % ndev) : 0;
    cudaSetDevice(dev);
    MPI_Comm_free(&local_comm);
    return dev;
}

extern "C" void fa2_forward_mpi(
    const float* Q_local_h,   // [Nq_local, d]
    const float* K_local_h,   // [Nk_local, d]
    const float* V_local_h,   // [Nk_local, dv]
    float* O_local_h,         // [Nq_local, dv]
    int Nq_local,
    int Nk_global,
    int d,
    int dv,
    float scale,
    int warps_per_block,
    MPI_Comm comm)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int base = Nk_global / size;
    int rem  = Nk_global % size;

    auto rows_of = [&](int r) -> int { return base + (r < rem ? 1 : 0); };
    auto off_of  = [&](int r) -> int { return r * base + (r < rem ? r : rem); };

    int Nk_local = rows_of(rank);

    std::vector<int> recvcountsK(size), displsK(size), recvcountsV(size), displsV(size);
    for (int r = 0; r < size; ++r) {
        int nkr = rows_of(r);
        int off = off_of(r);
        recvcountsK[r] = nkr * d;
        displsK[r]     = off * d;
        recvcountsV[r] = nkr * dv;
        displsV[r]     = off * dv;
    }

    std::vector<float> K_full_h((size_t)Nk_global * d);
    std::vector<float> V_full_h((size_t)Nk_global * dv);

    MPI_Allgatherv(
        K_local_h, Nk_local * d, MPI_FLOAT,
        K_full_h.data(), recvcountsK.data(), displsK.data(), MPI_FLOAT, comm);

    MPI_Allgatherv(
        V_local_h, Nk_local * dv, MPI_FLOAT,
        V_full_h.data(), recvcountsV.data(), displsV.data(), MPI_FLOAT, comm);

    bind_cuda_device_to_local_rank(comm);

    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    cudaMalloc(&dQ, (size_t)Nq_local * d  * sizeof(float));
    cudaMalloc(&dK, (size_t)Nk_global * d * sizeof(float));
    cudaMalloc(&dV, (size_t)Nk_global * dv* sizeof(float));
    cudaMalloc(&dO, (size_t)Nq_local * dv * sizeof(float));

    cudaMemcpy(dQ, Q_local_h, (size_t)Nq_local * d   * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K_full_h.data(), (size_t)Nk_global * d  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V_full_h.data(), (size_t)Nk_global * dv * sizeof(float), cudaMemcpyHostToDevice);

    fa2_fwd_kernel(dQ, dK, dV, dO, Nq_local, Nk_global, d, dv, scale, warps_per_block, 0);

    cudaDeviceSynchronize();
    cudaMemcpy(O_local_h, dO, (size_t)Nq_local * dv * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
}
