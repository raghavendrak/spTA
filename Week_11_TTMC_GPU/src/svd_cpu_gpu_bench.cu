#include <iostream>

// CUDA and cuSOLVER
extern "C" {
#include <cuda_runtime.h>
}
#include <cusolverDn.h>

// LAPACK (CPU SVD)
extern "C" {
#include <lapacke.h>
}

#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " code=" << static_cast<int>(err)                      \
                      << " \"" << cudaGetErrorString(err) << "\"\n";            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CHECK_CUSOLVER(call)                                                    \
    do {                                                                        \
        cusolverStatus_t status = (call);                                       \
        if (status != CUSOLVER_STATUS_SUCCESS) {                                \
            std::cerr << "cuSOLVER error at " << __FILE__ << ":" << __LINE__   \
                      << " status=" << static_cast<int>(status) << "\n";        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

int main() {
    // Matrix dimensions
    const int M = 60660;
    const int N = 30;
    const int NUM_NONZERO_ROWS = 264;

    // Random number generation
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Select 264 distinct random row indices in [0, M-1]
    std::vector<int> all_rows(M);
    for (int i = 0; i < M; ++i) all_rows[i] = i;
    std::shuffle(all_rows.begin(), all_rows.end(), rng);
    std::vector<int> nz_rows(all_rows.begin(), all_rows.begin() + NUM_NONZERO_ROWS);

    // Build matrix A in column-major layout (Fortran-style) for cuSOLVER
    std::vector<double> A(M * N, 0.0);
    for (int r : nz_rows) {
        for (int c = 0; c < N; ++c) {
            A[r + c * M] = dist(rng);
        }
    }

    std::cout << "Matrix size: " << M << " x " << N
              << ", non-zero rows: " << NUM_NONZERO_ROWS << "\n";

    const int min_mn = std::min(M, N);

    // -------------------
    // CPU SVD with LAPACKE (dgesdd – divide & conquer, efficient)
    // -------------------
    {
        // Copy A because LAPACKE_dgesdd overwrites its input
        std::vector<double> A_cpu = A;
        std::vector<double> S_cpu(min_mn);
        // Thin U: M x min_mn, thin VT: min_mn x N
        std::vector<double> U_cpu(M * min_mn);
        std::vector<double> VT_cpu(min_mn * N);

        auto t0 = std::chrono::high_resolution_clock::now();
        int info = LAPACKE_dgesdd(
            LAPACK_COL_MAJOR,  // column-major layout, matches A layout
            'S',               // compute thin U and VT
            M,
            N,
            A_cpu.data(),      // On entry: A; on exit: destroyed
            M,                 // lda
            S_cpu.data(),
            U_cpu.data(),
            M,                 // ldu
            VT_cpu.data(),
            min_mn             // ldvt (min(M,N) x N, leading dim = min(M,N))
        );
        auto t1 = std::chrono::high_resolution_clock::now();

        if (info != 0) {
            std::cerr << "CPU SVD (LAPACKE_dgesdd) failed, info = "
                      << info << "\n";
        } else {
            std::chrono::duration<double, std::milli> cpu_ms = t1 - t0;
            std::cout << "CPU SVD (LAPACKE dgesdd) time: "
                      << cpu_ms.count() << " ms\n";
            std::cout << "CPU SVD: largest singular value = "
                      << S_cpu[0] << "\n";
        }
    }

    // -------------------
    // GPU SVD with cuSOLVER
    // -------------------
    const int lda = M;

    cusolverDnHandle_t cusolverH = nullptr;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    double *d_A = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_A),
                          sizeof(double) * M * N));
    CHECK_CUDA(cudaMemcpy(d_A,
                          A.data(),
                          sizeof(double) * M * N,
                          cudaMemcpyHostToDevice));

    double *d_S = nullptr;
    double *d_U = nullptr;
    double *d_VT = nullptr;

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_S),
                          sizeof(double) * min_mn));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_U),
                          sizeof(double) * M * N));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_VT),
                          sizeof(double) * N * N));

    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnDgesvd_bufferSize(cusolverH, M, N, &lwork));

    double *d_work = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_work),
                          sizeof(double) * lwork));

    int *devInfo = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&devInfo),
                          sizeof(int)));

    signed char jobu = 'S';
    signed char jobvt = 'S';

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    double *d_A_work = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_A_work),
                          sizeof(double) * M * N));
    CHECK_CUDA(cudaMemcpy(d_A_work,
                          d_A,
                          sizeof(double) * M * N,
                          cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUSOLVER(
        cusolverDnDgesvd(
            cusolverH,
            jobu,
            jobvt,
            M,
            N,
            d_A_work,
            lda,
            d_S,
            d_U,
            lda,
            d_VT,
            N,
            d_work,
            lwork,
            nullptr,
            devInfo));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    int dev_info_h = 0;
    CHECK_CUDA(cudaMemcpy(&dev_info_h,
                          devInfo,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (dev_info_h != 0) {
        std::cerr << "GPU SVD: devInfo = " << dev_info_h
                  << " (SVD may not have converged)\n";
    }

    float gpu_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
    std::cout << "GPU SVD (cuSOLVER gesvd) time: " << gpu_ms
              << " ms (kernel only, data already on device)\n";

    std::vector<double> S_gpu(min_mn);
    CHECK_CUDA(cudaMemcpy(S_gpu.data(),
                          d_S,
                          sizeof(double) * min_mn,
                          cudaMemcpyDeviceToHost));
    std::cout << "GPU SVD: largest singular value = " << S_gpu[0] << "\n";

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_A_work));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_U));
    CHECK_CUDA(cudaFree(d_VT));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));

    return 0;
}

