// File: bcsf_mttkrp_gpu.cu
// Location: Week_11_TTMC_GPU/
// Build example:
//   nvcc -O3 -std=c++14 -arch=sm_60 bcsf_mttkrp_gpu.cu COO_to_CSF.cpp genten.c -o bcsf_mttkrp_gpu
//
// This is a self-contained B-CSF builder + GPU MTTKRP (mode-1) runner.
// It follows the same project style as your existing files and imports your CSF/COO utilities.
//
// References for B-CSF ideas (fiber splitting + slice splitting) and their GPU mapping:
//   Nisa et al., "Load-Balanced Sparse MTTKRP on GPUs", esp. Sec. IV and Fig. 2. (Attached PDF)
//
// Includes
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cassert>

#if defined(__has_include)
 #if __has_include("csf_tensor.h")
  #include "csf_tensor.h"
 #endif
 #if __has_include("matrix_utils.h")
  #include "matrix_utils.h"
 #endif
#endif

#include "genten.h"
#include "COO_to_CSF.h"

// ---------------------------------------------
// Helpers
// ---------------------------------------------
#define CUDA_OK(call)                                                            \
  do {                                                                           \
    cudaError_t _e = (call);                                                     \
    if (_e != cudaSuccess) {                                                     \
      std::cerr << "CUDA error " << cudaGetErrorName(_e) << " at "               \
                << __FILE__ << ":" << __LINE__ << " : "                          \
                << cudaGetErrorString(_e) << std::endl;                          \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static inline int64_t ceil_div(int64_t a, int64_t b)
{
  return (a + b - 1) / b;
}

static void fill_or_read_matrix(const char *fname, int64_t rows, int64_t cols, double *A)
{
  std::ifstream in(fname);
  if (in.good()) {
    for (int64_t i = 0; i < rows; i++) {
      for (int64_t j = 0; j < cols; j++) {
        if (!(in >> A[i * cols + j])) {
          A[i * cols + j] = 1.0;
        }
      }
    }
    return;
  }

  // fallback: ones
  for (int64_t i = 0; i < rows * cols; i++) {
    A[i] = 1.0;
  }
}

// ---------------------------------------------
// B-CSF host structure (3D)
// ---------------------------------------------
struct BCSF
{
  // Slices are sub-slices after slice-splitting; each sub-slice maps to one "i"
  int64_t  I, J, K;       // dimensions
  int64_t  nSlices;       // number of sub-slices
  int64_t  nFibers;       // fibers after fiber-splitting
  int64_t  nnz;           // total nnz (unchanged)

  // 1-based indices (consistent with your CSF builder)
  int64_t *slcPtr;        // size nSlices+1, points into fbr-range
  int64_t *slcIdx;        // size nSlices, stores i (1-based)
  int64_t *fbrPtr;        // size nFibers+1, points into kIdx/vals
  int64_t *fbrIdx;        // size nFibers, stores j (1-based)
  int64_t *kIdx;          // size nnz, stores k (1-based)
  double  *vals;          // size nnz

  BCSF()
  {
    I = J = K = 0;
    nSlices = nFibers = nnz = 0;
    slcPtr = slcIdx = fbrPtr = fbrIdx = kIdx = nullptr;
    vals = nullptr;
  }
};

// ---------------------------------------------
// Build B-CSF (fiber splitting + slice splitting) from CSF
// ---------------------------------------------
static void build_bcsf_from_csf(
  // CSF arrays (3D) from your COO_to_CSF conversion (1-based indices)
  const int64_t *mode_0_ptr, int size_mode_0_ptr,
  const int64_t *mode_0_idx, int size_mode_0_idx,
  const int64_t *mode_1_ptr, int size_mode_1_ptr,
  const int64_t *mode_1_idx, int size_mode_1_idx,
  const int64_t *mode_2_ptr, int size_mode_2_ptr,
  const int64_t *mode_2_idx, int size_mode_2_idx,
  const double  *vals_in,
  // dims
  int64_t I, int64_t J, int64_t K,
  // thresholds
  int64_t fiber_split_th,        // default 128
  int64_t slice_split_th,        // default blockSize (e.g., 256/512)
  // out
  BCSF &out)
{
  // CSF sanity (sizes) for 3D:  ptr0: S+1, idx0: S;  ptr1: F+1, idx1: F;  ptr2: M+1?, idx2: M
  // Our CSF builder stored level pointers as counts (ending at size of indices).
  // mode_2_ptr length == (#fibers + 1), mode_2_idx length == nnz.
  const int64_t S = size_mode_0_idx;
  const int64_t F = size_mode_1_idx;
  const int64_t M = size_mode_2_idx;

  out.I = I; out.J = J; out.K = K;
  out.nnz = M;

  // --------
  // Fiber splitting pass: compute split counts per original fiber
  // --------
  std::vector<int64_t> seg_count_per_f(F);
  int64_t F_new = 0;
  for (int64_t f = 0; f < F; f++) {
    const int64_t len = mode_2_ptr[f + 1] - mode_2_ptr[f];
    const int64_t seg = std::max<int64_t>(1, (int64_t)ceil_div(len, fiber_split_th));
    seg_count_per_f[f] = seg;
    F_new += seg;
  }
  out.nFibers = F_new;

  // --------
  // Build new fbrIdx/fbrPtr/kIdx/vals with fiber segments
  // --------
  out.fbrIdx = new int64_t[F_new];
  out.fbrPtr = new int64_t[F_new + 1];
  out.kIdx   = new int64_t[M];
  out.vals   = new double[M];

  int64_t fbr_cursor = 0;
  int64_t nz_cursor  = 0;

  for (int64_t f = 0; f < F; f++) {
    const int64_t j1b = mode_1_idx[f];              // 1-based j
    const int64_t z0  = mode_2_ptr[f];
    const int64_t z1  = mode_2_ptr[f + 1];
    const int64_t len = z1 - z0;

    if (len <= 0) {
      continue;
    }

    const int64_t seg = seg_count_per_f[f];
    const int64_t chunk = (int64_t)fiber_split_th;

    for (int64_t s = 0; s < seg; s++) {
      out.fbrIdx[fbr_cursor] = j1b;
      out.fbrPtr[fbr_cursor] = nz_cursor;

      const int64_t start = z0 + s * chunk;
      const int64_t end   = std::min<int64_t>(z0 + (s + 1) * chunk, z1);

      for (int64_t z = start; z < end; z++) {
        out.kIdx[nz_cursor] = mode_2_idx[z];
        out.vals[nz_cursor] = vals_in[z];
        nz_cursor++;
      }
      fbr_cursor++;
    }
  }
  out.fbrPtr[out.nFibers] = nz_cursor;

  // --------
  // Slice splitting:
  //   Split original slice s into Tb = ceil(nnz_in_slice / slice_split_th) sub-slices.
  //   We split by fiber-count proportional to Tb to keep things simple & close to the paper.
  // --------
  // First compute how many new fibers per original slice after fiber splitting
  std::vector<int64_t> fbr_count_per_slice(S, 0);
  {
    int64_t f_old = 0;
    int64_t f_new_prefix = 0;
    for (int64_t s = 0; s < S; s++) {
      const int64_t f_st = mode_1_ptr[s];
      const int64_t f_en = mode_1_ptr[s + 1];
      int64_t cnt = 0;
      for (int64_t f = f_st; f < f_en; f++) {
        cnt += seg_count_per_f[f];
      }
      fbr_count_per_slice[s] = cnt;
      f_old = f_en;
      f_new_prefix += cnt;
    }
    (void)f_old;
  }

  // Prefix sums of new fibers to map slice -> new-fiber (global) range
  std::vector<int64_t> new_fbr_psum(S + 1, 0);
  for (int64_t s = 0; s < S; s++) {
    new_fbr_psum[s + 1] = new_fbr_psum[s] + fbr_count_per_slice[s];
  }

  // Compute sub-slice counts and total sub-slices
  std::vector<int64_t> subs_per_slice(S, 0);
  int64_t S_new = 0;
  for (int64_t s = 0; s < S; s++) {
    // Estimate nnz in slice by summing fbr lengths quickly using new pointers
    const int64_t f_st_new = new_fbr_psum[s];
    const int64_t f_en_new = new_fbr_psum[s + 1];
    int64_t nnz_slc = 0;
    for (int64_t f2 = f_st_new; f2 < f_en_new; f2++) {
      nnz_slc += (out.fbrPtr[f2 + 1] - out.fbrPtr[f2]);
    }
    const int64_t Tb = std::max<int64_t>(1, (int64_t)ceil_div(nnz_slc, slice_split_th));
    subs_per_slice[s] = Tb;
    S_new += Tb;
  }
  out.nSlices = S_new;
  out.slcIdx = new int64_t[S_new];
  out.slcPtr = new int64_t[S_new + 1];

  // Partition each original slice’s new-fiber range into Tb nearly-equal chunks
  int64_t slc_cursor = 0;
  int64_t fbr_base   = 0;
  for (int64_t s = 0; s < S; s++) {
    const int64_t Tb = subs_per_slice[s];
    const int64_t f_st_new = new_fbr_psum[s];
    const int64_t f_en_new = new_fbr_psum[s + 1];
    const int64_t nF = f_en_new - f_st_new;

    if (Tb == 1) {
      out.slcIdx[slc_cursor]   = mode_0_idx[s];
      out.slcPtr[slc_cursor]   = f_st_new;
      slc_cursor++;
    }
    else {
      const int64_t chunk = (int64_t)ceil_div(nF, Tb);
      for (int64_t t = 0; t < Tb; t++) {
        const int64_t fb_st = f_st_new + t * chunk;
        const int64_t fb_en = std::min<int64_t>(f_st_new + (t + 1) * chunk, f_en_new);
        if (fb_st >= fb_en) {
          continue;
        }
        out.slcIdx[slc_cursor] = mode_0_idx[s];
        out.slcPtr[slc_cursor] = fb_st;
        slc_cursor++;
      }
    }
    fbr_base += fbr_count_per_slice[s];
  }
  out.slcPtr[out.nSlices] = out.nFibers;
}

// ---------------------------------------------
// GPU kernel: B-CSF MTTKRP (mode-1 update, i-mode output)
//   Each block processes one sub-slice.
//   Warps within a block process fibers (with stride warpPerSlice).
//   Lane-strided inner loop over rank R.
//   AtomicAdd to U0 because slice-splitting may create inter-block collisions.
// ---------------------------------------------
__global__ void mttkrp_bcsf_mode1_kernel(
  const int64_t * __restrict__ slcIdx,    // nSlices, i (1-based)
  const int64_t * __restrict__ slcPtr,    // nSlices+1 -> fiber range
  const int64_t * __restrict__ fbrPtr,    // nFibers+1 -> nnz range
  const int64_t * __restrict__ fbrIdx,    // nFibers, j (1-based)
  const int64_t * __restrict__ kIdx,      // nnz, k (1-based)
  const double  * __restrict__ vals,      // nnz
  double        * __restrict__ U0,        // I x R (output)
  const double  * __restrict__ U1,        // J x R (B)
  const double  * __restrict__ U2,        // K x R (C)
  int64_t nSlices,
  int      R,
  int      warpPerSlice)
{
  const int tId   = threadIdx.x;
  const int lane  = tId & 31;
  const int warp  = tId >> 5;
  const int warps = blockDim.x >> 5;

  const int64_t s = blockIdx.x;   // one sub-slice per block
  if (s >= nSlices) {
    return;
  }

  // 1-based i
  const int64_t i = slcIdx[s] - 1;

  const int64_t f_st = slcPtr[s];
  const int64_t f_en = slcPtr[s + 1];

  // stride across fibers by warp id
  for (int64_t f = f_st + warp; f < f_en; f += warpPerSlice) {

    const int64_t j = fbrIdx[f] - 1;
    const int64_t z0 = fbrPtr[f];
    const int64_t z1 = fbrPtr[f + 1];

    // lane-strided accumulation across rank
    for (int r = lane; r < R; r += 32) {
      double tmp = 0.0;

      // reduce across the k's in this fiber segment
      for (int64_t z = z0; z < z1; z++) {
        const int64_t k = kIdx[z] - 1;
        tmp += vals[z] * U2[k * R + r];
      }

      // multiply by B(j, :) and accumulate to A(i, :)
      const double bjr = U1[j * R + r];
      atomicAdd(&U0[i * R + r], tmp * bjr);
    }
  }
}

// ---------------------------------------------
// Host wrapper to run kernel
// ---------------------------------------------
static void run_mttkrp_bcsf_mode1_gpu(
  const BCSF &X,
  int R,
  const double *U1_host, const double *U2_host,
  double *U0_host,          // out, size I x R, must be zero-initialized
  int blockSize,
  int warpPerSlice)
{
  // Device allocations
  int64_t *d_slcPtr = nullptr, *d_slcIdx = nullptr;
  int64_t *d_fbrPtr = nullptr, *d_fbrIdx = nullptr;
  int64_t *d_kIdx   = nullptr;
  double  *d_vals   = nullptr;
  double  *d_U0     = nullptr;
  double  *d_U1     = nullptr;
  double  *d_U2     = nullptr;

  CUDA_OK(cudaMalloc(&d_slcPtr, (X.nSlices + 1) * sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_slcIdx, (X.nSlices)     * sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_fbrPtr, (X.nFibers + 1) * sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_fbrIdx, (X.nFibers)     * sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_kIdx,   (X.nnz)         * sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_vals,   (X.nnz)         * sizeof(double)));
  CUDA_OK(cudaMalloc(&d_U0,     (X.I * R)       * sizeof(double)));
  CUDA_OK(cudaMalloc(&d_U1,     (X.J * R)       * sizeof(double)));
  CUDA_OK(cudaMalloc(&d_U2,     (X.K * R)       * sizeof(double)));

  CUDA_OK(cudaMemcpy(d_slcPtr, X.slcPtr, (X.nSlices + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_slcIdx, X.slcIdx, (X.nSlices)     * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_fbrPtr, X.fbrPtr, (X.nFibers + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_fbrIdx, X.fbrIdx, (X.nFibers)     * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_kIdx,   X.kIdx,   (X.nnz)         * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_vals,   X.vals,   (X.nnz)         * sizeof(double),  cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_U0,     U0_host,  (X.I * R)       * sizeof(double),  cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_U1,     U1_host,  (X.J * R)       * sizeof(double),  cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_U2,     U2_host,  (X.K * R)       * sizeof(double),  cudaMemcpyHostToDevice));

  // Launch
  const dim3 grid((unsigned)X.nSlices, 1, 1);
  const dim3 block((unsigned)blockSize, 1, 1);

  mttkrp_bcsf_mode1_kernel<<<grid, block>>>(
    d_slcIdx, d_slcPtr, d_fbrPtr, d_fbrIdx, d_kIdx, d_vals,
    d_U0, d_U1, d_U2, X.nSlices, R, warpPerSlice);

  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());

  // Copy back
  CUDA_OK(cudaMemcpy(U0_host, d_U0, (X.I * R) * sizeof(double), cudaMemcpyDeviceToHost));

  // Cleanup
  cudaFree(d_slcPtr);
  cudaFree(d_slcIdx);
  cudaFree(d_fbrPtr);
  cudaFree(d_fbrIdx);
  cudaFree(d_kIdx);
  cudaFree(d_vals);
  cudaFree(d_U0);
  cudaFree(d_U1);
  cudaFree(d_U2);
}

// ---------------------------------------------
// Main: generate tensor -> CSF -> B-CSF -> MTTKRP (mode 1)
// ---------------------------------------------
int main(int argc, char **argv)
{
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <order> <I> <J> <K> <R> [ignored] [ignored]" << std::endl;
    return 1;
  }

  const int    order = std::atoi(argv[1]);
  const int64_t I = std::atoll(argv[2]);
  const int64_t J = std::atoll(argv[3]);
  const int64_t K = std::atoll(argv[4]);
  const int    R = std::atoi(argv[5]);

  if (order != 3) {
    std::cerr << "This file implements B-CSF for 3D tensors (order == 3)." << std::endl;
    return 1;
  }

  // Generate a COO tensor via genten (already in your project)
  int64_t *my_tensor_indices = nullptr;
  double  *my_tensor_values  = nullptr;
  int64_t total_indices = 0;
  int64_t total_values  = 0;

  generate_tensor(argc, argv, &my_tensor_indices, &my_tensor_values, &total_indices, &total_values);

  // Convert COO -> CSF using your existing utility
  cooToCSF(my_tensor_indices, my_tensor_values, order, total_indices, total_values);

  // Access CSF arrays produced by your converter
  int64_t *mode_0_ptr = nullptr, *mode_0_idx = nullptr;
  int64_t *mode_1_ptr = nullptr, *mode_1_idx = nullptr;
  int64_t *mode_2_ptr = nullptr, *mode_2_idx = nullptr;

  int sz_m0p=0, sz_m0i=0, sz_m1p=0, sz_m1i=0, sz_m2p=0, sz_m2i=0;
  get_mode_0_ptr(&mode_0_ptr, &sz_m0p);
  get_mode_0_idx(&mode_0_idx, &sz_m0i);
  get_mode_1_ptr(&mode_1_ptr, &sz_m1p);
  get_mode_1_idx(&mode_1_idx, &sz_m1i);
  get_mode_2_ptr(&mode_2_ptr, &sz_m2p);
  get_mode_2_idx(&mode_2_idx, &sz_m2i);

  // Dense factor matrices: B (JxR), C (KxR)
  std::vector<double> U1(J * R);
  std::vector<double> U2(K * R);
  fill_or_read_matrix("input_matrix_A.txt", J, R, U1.data());
  fill_or_read_matrix("input_matrix_B.txt", K, R, U2.data());

  // Output factor to update: A (IxR)
  std::vector<double> U0(I * R, 0.0);

  // Thresholds and kernel config (env overrides)
  const char *e_fth = std::getenv("BCSF_FBR_TH");
  const char *e_bsz = std::getenv("BCSF_BLOCK_SIZE");
  const char *e_wps = std::getenv("BCSF_WARPS_PER_SLICE");
  const char *e_sth = std::getenv("BCSF_SLC_TH");

  const int64_t fiber_split_th = e_fth ? std::atoll(e_fth) : 128;        // from paper evaluations
  const int     blockSize      = e_bsz ? std::atoi(e_bsz) : 256;
  const int     warpPerSlice   = e_wps ? std::atoi(e_wps) : std::max(1, blockSize/32);
  const int64_t slice_split_th = e_sth ? std::atoll(e_sth) : blockSize;  // paper suggests ~blockDim

  // Build B-CSF
  BCSF Xb;
  build_bcsf_from_csf(
    mode_0_ptr, sz_m0p, mode_0_idx, sz_m0i,
    mode_1_ptr, sz_m1p, mode_1_idx, sz_m1i,
    mode_2_ptr, sz_m2p, mode_2_idx, sz_m2i,
    my_tensor_values,
    I, J, K,
    fiber_split_th, slice_split_th,
    Xb);

  // Run GPU MTTKRP (mode-1) with timing
  auto t0 = std::chrono::high_resolution_clock::now();
  run_mttkrp_bcsf_mode1_gpu(Xb, R, U1.data(), U2.data(), U0.data(), blockSize, warpPerSlice);
  auto t1 = std::chrono::high_resolution_clock::now();
  double msec = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << "B-CSF MTTKRP (mode-1) completed in " << msec << " ms\n";

  // Optional: write result row sums for a quick sanity peek
  double checksum = 0.0;
  for (int64_t i = 0; i < I * R; i++) {
    checksum += U0[i];
  }
  std::cout << "Checksum(A): " << checksum << std::endl;

  // Clean up B-CSF host buffers
  delete [] Xb.slcPtr;
  delete [] Xb.slcIdx;
  delete [] Xb.fbrPtr;
  delete [] Xb.fbrIdx;
  delete [] Xb.kIdx;
  delete [] Xb.vals;

  // Free genten outputs
  free(my_tensor_indices);
  free(my_tensor_values);

  return 0;
}
