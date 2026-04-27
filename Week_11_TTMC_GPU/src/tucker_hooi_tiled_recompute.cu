/**
 * Tucker Decomposition via Higher-Order Orthogonal Iteration (HOOI)
 *
 * Reads a sparse tensor in COO format (.tns) directly, then builds 3 CSF copies
 * in memory (one per mode as root) for optimal compression at each TTMc step.
 * All TTMc operations use the single ncm_0 kernel with the appropriate CSF copy:
 *   - For TTMc skipping mode n, CSF copy n has mode n as root.
 *   - The remaining modes in each copy are sorted ascending by unique-index count
 *     (SPLATT-style compression).
 *
 * Usage:
 * nvcc -O3 -arch=sm_89 ./src/tucker_hooi.cu -o tucker_hooi.out -lcusolver -lcublas -lcudart
 * ./tucker_hooi.out [-v] [-c] -r 50 50 50 -t 0.01 -m 10 ./tns_datasets/nell-2.tns
 *
 * Flags:
 *   -v / --verbose   Print detailed per-step timing and SVD internals.
 *   -c / --check     Run CPU reference TTMc in iteration 0 and report error vs GPU.
 */

 #include <iostream>
 #include <fstream>
 #include <sstream>
 #include <vector>
 #include <array>
 #include <numeric>
 #include <algorithm>
 #include <filesystem>
 #include <cstring>
 #include <chrono>
 #include <stdexcept>
 #include <cmath>
 #include <random>
 #include <unordered_set>
 #include <limits>
 #include <cctype>
 #include <strings.h>
 #include <cuda_runtime.h>
 #include <cusolverDn.h>
 #include <cublas_v2.h>
#ifdef USE_MAGMA_EXACT_GPU
#include <magma_v2.h>
#include <magma_d.h>
#endif
 // #include "matrix_utils.h"
 
 
 // ===================================================================
 // Data type selection: change SCALAR_DOUBLE to 1 for FP64 (double).
 // ===================================================================
 #define SCALAR_DOUBLE 0  // 0 = float (FP32), 1 = double (FP64)
//  #define temp 1
 
 #if SCALAR_DOUBLE
   using scalar_t = double;
   #define cublasGemmT           cublasDgemm
   #define cublasScalT           cublasDscal
   #define cublasGeamT           cublasDgeam
   #define cublasNrm2T           cublasDnrm2
   #define cublasDgmmT           cublasDdgmm
   #define cusolverSyevdBufSizeT cusolverDnDsyevd_bufferSize
   #define cusolverSyevdT        cusolverDnDsyevd
   #define cusolverGesvdBufSizeT cusolverDnDgesvd_bufferSize
   #define cusolverGesvdT        cusolverDnDgesvd
   #define cublasSyrkT           cublasDsyrk
   #define cublasDasumT          cublasDasum
 #else
   using scalar_t = float;
   #define cublasGemmT           cublasSgemm
   #define cublasScalT           cublasSscal
   #define cublasGeamT           cublasSgeam
   #define cublasNrm2T           cublasSnrm2
   #define cublasDgmmT           cublasSdgmm
   #define cusolverSyevdBufSizeT cusolverDnSsyevd_bufferSize
   #define cusolverSyevdT        cusolverDnSsyevd
   #define cusolverGesvdBufSizeT cusolverDnSgesvd_bufferSize
   #define cusolverGesvdT        cusolverDnSgesvd
   #define cublasSyrkT           cublasSsyrk
   #define cublasDasumT          cublasSasum
 #endif
 
 #if SCALAR_DOUBLE
 const scalar_t eps_mach = 1.1102230246251565e-16; // DBL_EPSILON
#else
 const scalar_t eps_mach = 1.1920929e-7f;          // FLT_EPSILON
#endif

 // Bring std::cout/endl into scope for the v4 engine functions below.
 using std::cout;
 using std::endl;
 
 // ===================================================================
 // v4-Optimized TTMc Engine
 // Copied verbatim from ttmc_unified_best_try_v4_optimized_2.cu.
 // Do NOT modify run_ttmc_cuda() or any function it calls internally.
 // ===================================================================
 
 #define cudaCheckError(call)                                                        \
 do {                                                                                 \
     cudaError_t err = call;                                                          \
     if (err != cudaSuccess) {                                                        \
         std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " -> "        \
                   << cudaGetErrorString(err) << " (" << err << ")" << std::endl;      \
         std::exit(EXIT_FAILURE);                                                     \
     }                                                                                \
 } while (0)
 
 struct TaskRange {
   uint64_t i_ptr;
   uint64_t j_begin;
   uint64_t j_end;
   uint64_t k_begin;
   uint64_t k_end;
 };
 
inline int getEnvInt(const char* name, int default_value)
{
  const char* value = std::getenv(name);
  if (!value || !*value) return default_value;
  return atoi(value);
}

struct TensorOptimizationPolicy {
  std::string tensor_name;
  bool enable_pinned_blocked_full_svd = false;
  bool allow_iterative_eig_fallback = false;
  bool force_iterative_eig = false;
};

struct ExactTopRFullGramFallbackFailed : public std::runtime_error {
  ExactTopRFullGramFallbackFailed()
    : std::runtime_error("Exact top-R full-Gram fallback failed") {}
};

struct ExactTopRResolutionFailed : public std::runtime_error {
  ExactTopRResolutionFailed()
    : std::runtime_error("Exact top-R full-Gram resolution failed") {}
};

struct TiledGramResolutionFailed : public std::runtime_error {
  TiledGramResolutionFailed()
    : std::runtime_error("Tiled Gram resolution failed") {}
};

static bool g_force_gpu_iterative_eig = false;
static bool g_allow_gpu_iterative_eig_fallback = false;
static bool g_enable_pinned_blocked_full_svd = false;

static bool forceGpuIterativeEig()
{
  return g_force_gpu_iterative_eig;
}

static bool allowGpuIterativeEigFallback()
{
  return g_allow_gpu_iterative_eig_fallback;
}

static TensorOptimizationPolicy chooseTensorOptimizationPolicy(
  const std::string& tns_file,
  const std::vector<uint64_t>& ranks)
{
  TensorOptimizationPolicy policy;
  policy.tensor_name = std::filesystem::path(tns_file).filename().string();
  (void)ranks;

  // Legacy-first default. Add explicit tensor/rank allowlist entries here only
  // after auditing a concrete failing case.
  policy.enable_pinned_blocked_full_svd =
    getEnvInt("TTMC_ENABLE_PINNED_BLOCKED_FULL_SVD", 1) != 0;
  // Forced iterative eig routing is intentionally disabled. The iterative
  // solver is only allowed as a last-resort fallback after exact top-R fails.
  policy.force_iterative_eig = false;
  policy.allow_iterative_eig_fallback =
    (getEnvInt("TTMC_ALLOW_GPU_ITERATIVE_EIG_FALLBACK", 1) != 0);

  return policy;
}

inline bool useSyrkForGram(int cols)
{
  const int min_cols = std::max(1, getEnvInt("TTMC_SYRK_MIN_COLS", 512));
  return cols >= min_cols;
}

#if !SCALAR_DOUBLE
struct PinnedBlockedFullSVDPlan {
  bool active = false;
  bool use_syrk = false;
  int block_rows = 0;
  int lwork = 0;
  int full_svd_lwork = 0;
  int full_svd_min_mn = 0;
};

struct PinnedBlockedFullSVDWorkspace {
  double* d_gram_dp = nullptr;
  double* d_blk_dp = nullptr;
  double* d_W_dp = nullptr;
  double* d_work_dp = nullptr;
  scalar_t* d_full_s = nullptr;
  scalar_t* d_full_u = nullptr;
  scalar_t* d_full_colmaj_A = nullptr;
  scalar_t* d_full_work = nullptr;
  scalar_t* d_full_vt_dummy = nullptr;
  int* d_info = nullptr;
  uint64_t gram_elems = 0;
  uint64_t blk_elems = 0;
  uint64_t w_elems = 0;
  uint64_t work_elems = 0;
  uint64_t full_s_elems = 0;
  uint64_t full_u_elems = 0;
  uint64_t full_colmaj_elems = 0;
  uint64_t full_work_elems = 0;
};

static void freePinnedBlockedFullSVDWorkspace(PinnedBlockedFullSVDWorkspace& ws)
{
  if (ws.d_gram_dp) cudaFree(ws.d_gram_dp);
  if (ws.d_blk_dp) cudaFree(ws.d_blk_dp);
  if (ws.d_W_dp) cudaFree(ws.d_W_dp);
  if (ws.d_work_dp) cudaFree(ws.d_work_dp);
  if (ws.d_full_s) cudaFree(ws.d_full_s);
  if (ws.d_full_u) cudaFree(ws.d_full_u);
  if (ws.d_full_colmaj_A) cudaFree(ws.d_full_colmaj_A);
  if (ws.d_full_work) cudaFree(ws.d_full_work);
  if (ws.d_full_vt_dummy) cudaFree(ws.d_full_vt_dummy);
  if (ws.d_info) cudaFree(ws.d_info);
  ws = PinnedBlockedFullSVDWorkspace{};
}

static bool tryPreparePinnedBlockedFullSVDWorkspace(
  PinnedBlockedFullSVDWorkspace& ws,
  uint64_t gram_elems,
  uint64_t blk_elems,
  uint64_t w_elems,
  uint64_t work_elems,
  uint64_t full_s_elems,
  uint64_t full_u_elems,
  uint64_t full_colmaj_elems,
  uint64_t full_work_elems)
{
  freePinnedBlockedFullSVDWorkspace(ws);
  if (gram_elems == 0 || blk_elems == 0 || w_elems == 0 || work_elems == 0) return true;

  cudaError_t err = cudaMalloc(&ws.d_gram_dp, sizeof(double) * gram_elems);
  if (err != cudaSuccess) { cudaGetLastError(); freePinnedBlockedFullSVDWorkspace(ws); return false; }

  err = cudaMalloc(&ws.d_blk_dp, sizeof(double) * blk_elems);
  if (err != cudaSuccess) { cudaGetLastError(); freePinnedBlockedFullSVDWorkspace(ws); return false; }

  err = cudaMalloc(&ws.d_W_dp, sizeof(double) * w_elems);
  if (err != cudaSuccess) { cudaGetLastError(); freePinnedBlockedFullSVDWorkspace(ws); return false; }

  err = cudaMalloc(&ws.d_work_dp, sizeof(double) * work_elems);
  if (err != cudaSuccess) { cudaGetLastError(); freePinnedBlockedFullSVDWorkspace(ws); return false; }

  if (full_s_elems > 0) {
    err = cudaMalloc(&ws.d_full_s, sizeof(scalar_t) * full_s_elems);
    if (err != cudaSuccess) { cudaGetLastError(); freePinnedBlockedFullSVDWorkspace(ws); return false; }
  }
  if (full_u_elems > 0) {
    err = cudaMalloc(&ws.d_full_u, sizeof(scalar_t) * full_u_elems);
    if (err != cudaSuccess) { cudaGetLastError(); freePinnedBlockedFullSVDWorkspace(ws); return false; }
  }
  if (full_colmaj_elems > 0) {
    err = cudaMalloc(&ws.d_full_colmaj_A, sizeof(scalar_t) * full_colmaj_elems);
    if (err != cudaSuccess) { cudaGetLastError(); freePinnedBlockedFullSVDWorkspace(ws); return false; }
  }
  if (full_work_elems > 0) {
    err = cudaMalloc(&ws.d_full_work, sizeof(scalar_t) * full_work_elems);
    if (err != cudaSuccess) { cudaGetLastError(); freePinnedBlockedFullSVDWorkspace(ws); return false; }
  }
  err = cudaMalloc(&ws.d_full_vt_dummy, sizeof(scalar_t));
  if (err != cudaSuccess) { cudaGetLastError(); freePinnedBlockedFullSVDWorkspace(ws); return false; }

  err = cudaMalloc(&ws.d_info, sizeof(int));
  if (err != cudaSuccess) { cudaGetLastError(); freePinnedBlockedFullSVDWorkspace(ws); return false; }

  ws.gram_elems = gram_elems;
  ws.blk_elems = blk_elems;
  ws.w_elems = w_elems;
  ws.work_elems = work_elems;
  ws.full_s_elems = full_s_elems;
  ws.full_u_elems = full_u_elems;
  ws.full_colmaj_elems = full_colmaj_elems;
  ws.full_work_elems = full_work_elems;
  return true;
}

static void gpu_truncated_svd_update_factor_pinned_blocked(
  cusolverDnHandle_t cusolverH,
  cublasHandle_t cublasH,
  scalar_t* d_A,
  int M,
  int N,
  int R,
  scalar_t* d_factor,
  const PinnedBlockedFullSVDPlan& plan,
  PinnedBlockedFullSVDWorkspace& ws,
  bool verbose);

void gpu_full_svd_update_factor(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH,
  scalar_t* d_A, int M, int N, int R, scalar_t* d_factor, bool verbose);

static bool tryLatePinnedBlockedFullSVDFallback(
  cusolverDnHandle_t cusolverH,
  cublasHandle_t cublasH,
  scalar_t* d_A,
  int M,
  int N,
  int R,
  scalar_t* d_factor,
  bool verbose)
{
  if (!(M > N) || N <= 0) return false;

  PinnedBlockedFullSVDPlan plan;
  plan.use_syrk = useSyrkForGram(N);
  plan.block_rows = std::min(std::max(1, getEnvInt("TTMC_BLOCKED_DP_ROWS", 32768)), M);
  plan.full_svd_min_mn = N;

  double* d_blk_probe = nullptr;
  while (plan.block_rows > 0) {
    cudaError_t blk_err =
      cudaMalloc(&d_blk_probe, sizeof(double) * (long long)plan.block_rows * N);
    if (blk_err == cudaSuccess) break;
    cudaGetLastError();
    if (verbose) {
      std::cout << "  late pinned blocked full-SVD probe failed for "
                << plan.block_rows << " rows, retrying smaller block\n";
    }
    plan.block_rows /= 2;
  }
  if (!d_blk_probe) {
    if (verbose) {
      std::cout << "  late pinned blocked full-SVD could not allocate any blocked Gram workspace\n";
    }
    return false;
  }
  cudaError_t free_err = cudaFree(d_blk_probe);
  if (free_err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }

  double* d_gram_probe = nullptr;
  double* d_w_probe = nullptr;
  double* d_work_probe = nullptr;
  auto free_probes = [&]() {
    if (d_work_probe) cudaFree(d_work_probe);
    if (d_w_probe) cudaFree(d_w_probe);
    if (d_gram_probe) cudaFree(d_gram_probe);
  };

  cudaError_t err = cudaMalloc(&d_gram_probe, sizeof(double) * (long long)N * N);
  if (err != cudaSuccess) {
    cudaGetLastError();
    if (verbose) std::cout << "  late pinned blocked full-SVD Gram probe alloc failed\n";
    return false;
  }
  err = cudaMalloc(&d_w_probe, sizeof(double) * N);
  if (err != cudaSuccess) {
    cudaGetLastError();
    free_probes();
    if (verbose) std::cout << "  late pinned blocked full-SVD eigenvalue probe alloc failed\n";
    return false;
  }

  int lwork = 0;
  cusolverStatus_t st = cusolverDnDsyevd_bufferSize(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER, N, d_gram_probe, N, d_w_probe, &lwork);
  if (st != CUSOLVER_STATUS_SUCCESS) {
    free_probes();
    if (verbose) {
      std::cout << "  late pinned blocked full-SVD eigensolver bufferSize failed with status="
                << static_cast<int>(st) << "\n";
    }
    return false;
  }

  err = cudaMalloc(&d_work_probe, sizeof(double) * std::max(lwork, 1));
  if (err != cudaSuccess) {
    cudaGetLastError();
    free_probes();
    if (verbose) std::cout << "  late pinned blocked full-SVD eigensolver work probe alloc failed\n";
    return false;
  }
  free_probes();

  int full_svd_lwork = 0;
  cusolverStatus_t full_svd_st = cusolverGesvdBufSizeT(cusolverH, M, N, &full_svd_lwork);
  if (full_svd_st != CUSOLVER_STATUS_SUCCESS && verbose) {
    std::cout << "  late pinned blocked full-SVD gesvd bufferSize failed with status="
              << static_cast<int>(full_svd_st)
              << ", using local workspace if pinned full SVD is reached\n";
  }

  plan.lwork = std::max(lwork, 1);
  plan.full_svd_lwork =
    (full_svd_st == CUSOLVER_STATUS_SUCCESS) ? std::max(full_svd_lwork, 1) : 0;
  plan.active = true;

  PinnedBlockedFullSVDWorkspace ws;
  bool ws_ok = tryPreparePinnedBlockedFullSVDWorkspace(
    ws,
    (uint64_t)N * N,
    (uint64_t)plan.block_rows * N,
    (uint64_t)N,
    (uint64_t)plan.lwork,
    (uint64_t)plan.full_svd_min_mn,
    (uint64_t)M * plan.full_svd_min_mn,
    (uint64_t)M * N,
    (uint64_t)plan.full_svd_lwork);
  if (!ws_ok) {
    if (verbose) {
      std::cout << "  late pinned blocked full-SVD workspace allocation failed\n";
    }
    return false;
  }

  try {
    gpu_truncated_svd_update_factor_pinned_blocked(
      cusolverH, cublasH, d_A, M, N, R, d_factor, plan, ws, verbose);
  } catch (...) {
    freePinnedBlockedFullSVDWorkspace(ws);
    throw;
  }

  freePinnedBlockedFullSVDWorkspace(ws);
  return true;
}

static void runSVdStyleRescueFromFullMatrix(
  cusolverDnHandle_t cusolverH,
  cublasHandle_t cublasH,
  scalar_t* d_A,
  int M,
  int N,
  int R,
  scalar_t* d_factor,
  bool verbose)
{
  bool rescued = false;
  if (g_enable_pinned_blocked_full_svd && !forceGpuIterativeEig() &&
      M > N && N > 0) {
    if (verbose) {
      std::cout << "  trying SVD-style rescue via late pinned blocked full-SVD\n";
    }
    rescued = tryLatePinnedBlockedFullSVDFallback(
      cusolverH, cublasH, d_A, M, N, R, d_factor, verbose);
  }
  if (!rescued) {
    if (verbose) {
      std::cout << "  trying SVD-style rescue via regular full SVD\n";
    }
    gpu_full_svd_update_factor(cusolverH, cublasH, d_A, M, N, R, d_factor, verbose);
  }
}
#endif

 inline bool getEnvFlag(const char* name)
 {
   const char* value = std::getenv(name);
   if (!value) return false;
   if (*value == '\0') return false;
   if (strcmp(value, "0") == 0) return false;
   if (strcasecmp(value, "false") == 0) return false;
   if (strcasecmp(value, "off") == 0) return false;
   return true;
 }
 
 struct FiberStats {
   uint64_t num_i = 0;
   uint64_t min_k_per_j = 0;
   uint64_t max_k_per_j = 0;
   double avg_k_per_j = 0.0;
   uint64_t total_j_fibers = 0;
 };
 
 struct DynamicHints {
   uint64_t base_tile = 64;
   uint64_t k_tile = 8192;
   int dynamic_block_hint = 256;
   int grid_factor_hint = 8;
 };
 
 static FiberStats analyzeFiberStats(
   const uint64_t* mode_1_ptr,
   const uint64_t* mode_2_ptr,
   uint64_t num_i)
 {
   FiberStats stats;
   stats.num_i = num_i;
   uint64_t min_k = UINT64_MAX;
   uint64_t max_k = 0;
   long double sum_k = 0.0L;
   uint64_t total_j = 0;
 
   for (uint64_t i_ptr = 0; i_ptr < num_i; ++i_ptr) {
     uint64_t begin = mode_1_ptr[i_ptr];
     uint64_t end = mode_1_ptr[i_ptr + 1];
     if (begin >= end) continue;
 
     for (uint64_t j_ptr = begin; j_ptr < end; ++j_ptr) {
       uint64_t k_begin = mode_2_ptr[j_ptr];
       uint64_t k_end = mode_2_ptr[j_ptr + 1];
       uint64_t k_len = k_end - k_begin;
       min_k = std::min(min_k, k_len);
       max_k = std::max(max_k, k_len);
       sum_k += k_len;
       ++total_j;
     }
   }
 
   if (min_k == UINT64_MAX) min_k = 0;
   stats.min_k_per_j = min_k;
   stats.max_k_per_j = max_k;
   stats.total_j_fibers = total_j;
   stats.avg_k_per_j = (total_j > 0)
     ? static_cast<double>(sum_k) / static_cast<double>(total_j)
     : 0.0;
   return stats;
 }
 
 static DynamicHints chooseDynamicHints(double avg_k_per_j, bool force_dynamic)
 {
   DynamicHints hints;
   bool use_large_tile = (avg_k_per_j <= 12.0);
   if (use_large_tile && !force_dynamic) {
     hints.base_tile = 128;
   }
   hints.dynamic_block_hint = use_large_tile ? 416 : 256;
   hints.grid_factor_hint = use_large_tile ? 12 : 8;
   hints.dynamic_block_hint = getEnvInt("TTMC_DYNAMIC_BLOCK", hints.dynamic_block_hint);
   if (hints.dynamic_block_hint < 32) hints.dynamic_block_hint = 32;
   hints.dynamic_block_hint = (hints.dynamic_block_hint / 32) * 32;
   if (hints.dynamic_block_hint == 0) hints.dynamic_block_hint = 32;
 
   hints.grid_factor_hint = getEnvInt("TTMC_DYNAMIC_GRID_FACTOR", hints.grid_factor_hint);
   if (hints.grid_factor_hint < 1) hints.grid_factor_hint = 1;
 
   int k_tile_hint = getEnvInt("TTMC_DYNAMIC_K_TILE", 0);
   if (k_tile_hint > 0) {
     hints.k_tile = static_cast<uint64_t>(k_tile_hint);
   }
   else {
     if (avg_k_per_j > 6000.0) {
       hints.k_tile = 2048;
     }
     else if (avg_k_per_j > 2500.0) {
       hints.k_tile = 1024;
     }
     else if (avg_k_per_j > 900.0) {
       hints.k_tile = 512;
     }
     else if (avg_k_per_j > 300.0) {
       hints.k_tile = 256;
     }
   }
   if (hints.k_tile < 256) hints.k_tile = 256;
   return hints;
 }
 
 static void buildDynamicTaskRanges(
   const uint64_t* mode_1_ptr,
   const uint64_t* mode_2_ptr,
   uint64_t num_i,
   uint64_t base_tile,
   uint64_t k_tile,
   std::vector<TaskRange>& host_tasks,
   size_t reserve_hint)
 {
   host_tasks.clear();
   if (reserve_hint > 0) {
     host_tasks.reserve(reserve_hint);
   }
   for (uint64_t i_ptr = 0; i_ptr < num_i; ++i_ptr) {
     uint64_t begin = mode_1_ptr[i_ptr];
     uint64_t end = mode_1_ptr[i_ptr + 1];
     if (begin >= end) continue;
     uint64_t pending_start = begin;
     for (uint64_t j_ptr = begin; j_ptr < end; ++j_ptr) {
       uint64_t k_begin = mode_2_ptr[j_ptr];
       uint64_t k_end = mode_2_ptr[j_ptr + 1];
       uint64_t k_len = k_end - k_begin;
       if (k_len > k_tile) {
         if (pending_start < j_ptr) {
           for (uint64_t chunk = pending_start; chunk < j_ptr; chunk += base_tile) {
             TaskRange task;
             task.i_ptr = i_ptr;
             task.j_begin = chunk;
             task.j_end = std::min<uint64_t>(chunk + base_tile, j_ptr);
             task.k_begin = 0;
             task.k_end = 0;
             host_tasks.push_back(task);
           }
         }
         for (uint64_t k_chunk = k_begin; k_chunk < k_end; k_chunk += k_tile) {
           TaskRange task;
           task.i_ptr = i_ptr;
           task.j_begin = j_ptr;
           task.j_end = j_ptr + 1;
           task.k_begin = k_chunk;
           task.k_end = std::min<uint64_t>(k_chunk + k_tile, k_end);
           host_tasks.push_back(task);
         }
         pending_start = j_ptr + 1;
       }
     }
     if (pending_start < end) {
       for (uint64_t chunk = pending_start; chunk < end; chunk += base_tile) {
         TaskRange task;
         task.i_ptr = i_ptr;
         task.j_begin = chunk;
         task.j_end = std::min<uint64_t>(chunk + base_tile, end);
         task.k_begin = 0;
         task.k_end = 0;
         host_tasks.push_back(task);
       }
     }
   }
 }
 
 static size_t computeTaskReserveHint(uint64_t mode_1_size)
 {
   if (mode_1_size >= static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
     return 0;
   }
   size_t reserve_hint = static_cast<size_t>(mode_1_size);
   size_t reserve_cap = static_cast<size_t>(std::numeric_limits<size_t>::max() / 4);
   reserve_hint = std::min(reserve_hint + static_cast<size_t>(1024), reserve_cap);
   return reserve_hint;
 }
 
 static void allocateTaskBuffers(
   const std::vector<TaskRange>& host_tasks,
   TaskRange** d_tasks,
   unsigned long long** d_task_counter)
 {
   *d_tasks = nullptr;
   *d_task_counter = nullptr;
   cudaCheckError(cudaMalloc(d_task_counter, sizeof(unsigned long long)));
   cudaCheckError(cudaMemset(*d_task_counter, 0, sizeof(unsigned long long)));
   if (!host_tasks.empty()) {
     cudaCheckError(cudaMalloc(d_tasks, host_tasks.size() * sizeof(TaskRange)));
     cudaCheckError(cudaMemcpy(
       *d_tasks,
       host_tasks.data(),
       host_tasks.size() * sizeof(TaskRange),
       cudaMemcpyHostToDevice));
   }
 }
 
 static void freeTaskBuffers(TaskRange* d_tasks, unsigned long long* d_task_counter)
 {
   if (d_tasks) cudaFree(d_tasks);
   if (d_task_counter) cudaFree(d_task_counter);
 }
 
 // ===================================================================
 // TTMcCache: stores HOOI-loop-invariant preprocessing results.
 // Call prepare_ttmc_cuda() once per CSF copy before the HOOI loop,
 // then pass the filled cache to every run_ttmc_cuda() call.
 // ===================================================================
 struct TTMcCache {
   bool initialized      = false;
   int  order            = 0;
   // Path selection
   bool prefer_static        = false;  // 3D: kernel_ttmc3_static_block_per_i
   bool prefer_tiny_streams  = false;  // 4D: launch_ttmc4_tiny_streams_flat
   // 3D static-block-per-i params
   int    s_block_size = 0;
   int    s_num_warps  = 0;
   size_t s_shared_mem = 0;
   int    s_grid_size  = 0;
   // 3D / 4D dynamic-tasks params
   std::vector<TaskRange>  host_tasks;
   TaskRange*              d_tasks        = nullptr;  // persistent GPU task list
   unsigned long long*     d_task_counter = nullptr;  // persistent; reset before each launch
   int    d_block_size = 0;
   int    d_grid_size  = 0;
   size_t d_shared_mem = 0;
   // 4D tiny-streams params
   int tiny_stream_count = 0;
   // HOST arrays (non-owning ptrs into CSFCopy.ptrs/idxs) for tiny-streams launch loop.
   std::vector<uint64_t*> h_mode_ptrs;  // HOST: ptrs[l].data() for each level
   std::vector<uint64_t*> h_mode_idxs;  // HOST: idxs[l].data() for each level
   // Cached array sizes (invariant across HOOI iterations)
   std::vector<uint64_t>  size_mode_ptr;   // ptrs[l].size() per level
   std::vector<uint64_t>  size_mode_idx;   // idxs[l].size() per level
   std::vector<uint64_t>  factor_sizes;    // CSF-level indexed factor matrix sizes
 };
 
 void free_ttmc_cache(TTMcCache& c) {
   if (c.d_tasks)        { cudaFree(c.d_tasks);        c.d_tasks        = nullptr; }
   if (c.d_task_counter) { cudaFree(c.d_task_counter); c.d_task_counter = nullptr; }
   c.host_tasks.clear();
   c.h_mode_ptrs.clear();
   c.h_mode_idxs.clear();
   c.size_mode_ptr.clear();
   c.size_mode_idx.clear();
   c.factor_sizes.clear();
   c.initialized = false;
 }
 
 // Forward declarations needed by prepare_ttmc_cuda
 __global__ void kernel_ttmc3_static_block_per_i(
   const uint64_t*, const uint64_t*, const uint64_t*,
   const uint64_t*, const uint64_t*,
   const scalar_t*, scalar_t*, scalar_t*, scalar_t*,
   uint32_t, uint32_t, int);
 __global__ void kernel_ttmc3_dynamic_tasks(
   const uint64_t*, const uint64_t*, const uint64_t*,
   const uint64_t*, const uint64_t*,
   const scalar_t*, const scalar_t*, const scalar_t*,
   scalar_t*, uint32_t, uint32_t,
   const TaskRange*, uint64_t, unsigned long long*);
 __global__ void kernel_ttmc4_dynamic_tasks(
   const uint64_t*, const uint64_t*, const uint64_t*,
   const uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*,
   const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*,
   scalar_t*, uint32_t, uint32_t, uint32_t,
   const TaskRange*, uint64_t, unsigned long long*);
 
 // Perform all HOOI-loop-invariant preprocessing for one CSF copy.
 // mode_ptrs[i]  = HOST ptr to ptrs[i]; mode_idxs[i] = HOST ptr to idxs[i]
 // ranks          = CSF-level indexed ranks; factor_sizes = CSF-level indexed factor sizes
 // dimensions     = dimension of each CSF level
 // ncm            = non-contracting mode index (always 0 in HOOI)
 void prepare_ttmc_cuda(
   uint64_t** mode_ptrs, uint64_t** mode_idxs,
   uint64_t* size_mode_ptr, uint64_t* size_mode_idx,
   uint64_t* ranks, uint64_t* factor_sizes_in,
   uint64_t* dimensions,
   int ncm, int order,
   TTMcCache& cache)
 {
   free_ttmc_cache(cache);
   cache.order = order;
   // Cache HOST pointers and sizes (for launch_ttmc4_tiny_streams_flat)
   cache.h_mode_ptrs.assign(mode_ptrs,      mode_ptrs      + order);
   cache.h_mode_idxs.assign(mode_idxs,      mode_idxs      + order);
   cache.size_mode_ptr.assign(size_mode_ptr, size_mode_ptr  + order);
   cache.size_mode_idx.assign(size_mode_idx, size_mode_idx  + order);
   cache.factor_sizes.assign(factor_sizes_in, factor_sizes_in + order);
 
   std::vector<int> other_modes;
   for (int m = 0; m < order; ++m)
     if (m != ncm) other_modes.push_back(m);
   int idx_A = other_modes[0];
   int idx_B = other_modes[1];
   int idx_C = ((int)other_modes.size() > 2) ? other_modes[2] : -1;
   int f1 = (int)ranks[idx_A];
   int f2 = (int)ranks[idx_B];
   int f3 = (idx_C >= 0) ? (int)ranks[idx_C] : 0;
   uint64_t num_i = size_mode_ptr[1] - 1;
 
   cudaCheckError(cudaMalloc(&cache.d_task_counter, sizeof(unsigned long long)));
   cudaCheckError(cudaMemset(cache.d_task_counter, 0, sizeof(unsigned long long)));
 
   int device_id = 0;
   cudaCheckError(cudaGetDevice(&device_id));
   cudaDeviceProp prop{};
   cudaCheckError(cudaGetDeviceProperties(&prop, device_id));
 
   if (order == 3 && ncm == 0) {
     bool force_static  = getEnvFlag("TTMC_FORCE_STATIC");
     bool force_dynamic = getEnvFlag("TTMC_FORCE_DYNAMIC");
     FiberStats   stats = analyzeFiberStats(mode_ptrs[1], mode_ptrs[2], num_i);
     DynamicHints hints = chooseDynamicHints(stats.avg_k_per_j, force_dynamic);
     uint64_t base_tile     = hints.base_tile;
     uint64_t k_tile        = hints.k_tile;
     int dyn_block          = hints.dynamic_block_hint;
     int grid_factor        = hints.grid_factor_hint;
 
     bool ultra_tiny_k = (stats.max_k_per_j <= 5 && stats.avg_k_per_j <= 1.10);
     bool long_tail    = (stats.avg_k_per_j <= 12.0 && stats.max_k_per_j >= 15000 &&
                          dimensions[0] <= 700000);
     bool med_dyn      = (stats.avg_k_per_j >= 300.0 && stats.avg_k_per_j <= 1500.0 &&
                          stats.max_k_per_j <= 2000 && num_i <= 512);
     if (!force_dynamic && !force_static && med_dyn) {
       if (k_tile < 512) k_tile = 512;
       dyn_block = 128;
       if (grid_factor < 12) grid_factor = 12;
       base_tile = 64;
     }
     bool prefer_static = force_static || (!force_dynamic && (ultra_tiny_k || long_tail));
     if (f1 > 64) prefer_static = false;
     cache.prefer_static = prefer_static;
 
     if (prefer_static) {
       int def_blk   = (ultra_tiny_k || long_tail) ? 768 : 1024;
       int block_sz  = getEnvInt("TTMC_BLOCK_PER_I_BLOCK", def_blk);
       if (block_sz < 32) block_sz = 32;
       block_sz = (block_sz / 32) * 32;
       if (block_sz == 0) block_sz = 32;
       int nw = std::max(1, block_sz / 32);
       cache.s_block_size = block_sz;
       cache.s_num_warps  = nw;
       cache.s_shared_mem = (size_t)nw * f2 * sizeof(scalar_t) +
                            (size_t)f1 * f2 * sizeof(scalar_t);
       cache.s_grid_size  = (int)size_mode_idx[0];
     } else {
       buildDynamicTaskRanges(mode_ptrs[1], mode_ptrs[2], num_i, base_tile, k_tile,
                              cache.host_tasks, computeTaskReserveHint(size_mode_idx[1]));
       int blk = dyn_block, wpb = blk / 32;
       bool use_reg = (f1 <= 64 && f2 <= 32);
       size_t sh = use_reg ? 0 :
         (size_t)wpb * f2 * sizeof(scalar_t) +
         (size_t)wpb * (size_t)f1 * f2 * sizeof(scalar_t);
       size_t def_sh = prop.sharedMemPerBlock;
       size_t max_sh = (prop.sharedMemPerBlockOptin > def_sh) ? prop.sharedMemPerBlockOptin : def_sh;
       while (!use_reg && sh > max_sh && blk > 32) {
         blk -= 32; wpb = blk / 32;
         sh = (size_t)wpb * f2 * sizeof(scalar_t) +
              (size_t)wpb * (size_t)f1 * f2 * sizeof(scalar_t);
       }
       if (wpb == 0) { blk = 32; wpb = 1;
         sh = use_reg ? 0 : (size_t)f2 * sizeof(scalar_t) + (size_t)f1 * f2 * sizeof(scalar_t); }
       if (sh > max_sh) sh = max_sh;
       if (sh > def_sh && prop.sharedMemPerBlockOptin > def_sh)
         cudaCheckError(cudaFuncSetAttribute(kernel_ttmc3_dynamic_tasks,
           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)std::min(sh, max_sh)));
       else
         cudaCheckError(cudaFuncSetAttribute(kernel_ttmc3_dynamic_tasks,
           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)def_sh));
       if (!cache.host_tasks.empty()) {
         cudaCheckError(cudaMalloc(&cache.d_tasks, cache.host_tasks.size() * sizeof(TaskRange)));
         cudaCheckError(cudaMemcpy(cache.d_tasks, cache.host_tasks.data(),
           cache.host_tasks.size() * sizeof(TaskRange), cudaMemcpyHostToDevice));
       }
       cache.d_block_size = blk;
       cache.d_grid_size  = std::max(prop.multiProcessorCount * grid_factor, 1);
       cache.d_shared_mem = sh;
     }
   }
   else if (order == 4 && ncm == 0) {
     bool force_dyn   = getEnvFlag("TTMC_FORCE_DYNAMIC");
     bool force_tiny  = getEnvFlag("TTMC_FORCE_TINY_STREAMS");
     bool dis_tiny    = getEnvFlag("TTMC_DISABLE_TINY_STREAMS");
     FiberStats stats = analyzeFiberStats(mode_ptrs[1], mode_ptrs[2], num_i);
     double avg_k = stats.avg_k_per_j;
     uint64_t max_dim = 0;
     for (int d = 0; d < order; ++d) max_dim = std::max(max_dim, dimensions[d]);
     uint64_t total_nnz = size_mode_idx[order - 1];
     int td = std::max(1, getEnvInt("TTMC_TINY4D_MAX_DIM",      500000));
     int tn = std::max(1, getEnvInt("TTMC_TINY4D_MAX_NNZ",    60000000));
     int ta = std::max(1, getEnvInt("TTMC_TINY4D_MAX_AVG_K",       64));
     int sd = std::max(1, getEnvInt("TTMC_TINY4D_SMALL_DIM",    20000));
     int sn = std::max(1, getEnvInt("TTMC_TINY4D_SMALL_NNZ", 10000000));
     bool prefer_tiny = false;
     if (idx_C >= 0) {
       if (force_tiny) prefer_tiny = true;
       else if (!force_dyn && !dis_tiny) {
         bool under_dim = max_dim   <= (uint64_t)td;
         bool under_nnz = total_nnz <= (uint64_t)tn;
         bool under_avg = avg_k     <= (double)ta;
         bool very_small= max_dim   <= (uint64_t)sd && total_nnz <= (uint64_t)sn;
         prefer_tiny = under_dim && under_nnz && (under_avg || very_small);
       }
     }
     cache.prefer_tiny_streams = prefer_tiny;
     if (prefer_tiny) {
       int def_s = (int)std::min(num_i, (uint64_t)32);
       if (def_s <= 0) def_s = 1;
       cache.tiny_stream_count = getEnvInt("TTMC_TINY4D_NUM_STREAMS", def_s);
       if (cache.tiny_stream_count <= 0) cache.tiny_stream_count = def_s;
     } else {
       DynamicHints hints = chooseDynamicHints(avg_k, force_dyn);
       buildDynamicTaskRanges(mode_ptrs[1], mode_ptrs[2], num_i,
                              hints.base_tile, hints.k_tile,
                              cache.host_tasks, computeTaskReserveHint(size_mode_idx[1]));
       int blk = hints.dynamic_block_hint, wpb = blk / 32;
       size_t sh = (size_t)wpb * ((size_t)f2 * f3 + f2) * sizeof(scalar_t);
       size_t def_sh = prop.sharedMemPerBlock;
       size_t max_sh = (prop.sharedMemPerBlockOptin > def_sh) ? prop.sharedMemPerBlockOptin : def_sh;
       while (sh > max_sh && blk > 32) {
         blk -= 32; wpb = blk / 32;
         sh = (size_t)wpb * ((size_t)f2 * f3 + f2) * sizeof(scalar_t);
       }
       if (wpb == 0) { blk = 32; wpb = 1;
         sh = (size_t)wpb * ((size_t)f2 * f3 + f2) * sizeof(scalar_t); }
       if (sh > max_sh) sh = max_sh;
       cudaCheckError(cudaFuncSetAttribute(kernel_ttmc4_dynamic_tasks,
         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)std::min(sh, max_sh)));
       if (!cache.host_tasks.empty()) {
         cudaCheckError(cudaMalloc(&cache.d_tasks, cache.host_tasks.size() * sizeof(TaskRange)));
         cudaCheckError(cudaMemcpy(cache.d_tasks, cache.host_tasks.data(),
           cache.host_tasks.size() * sizeof(TaskRange), cudaMemcpyHostToDevice));
       }
       cache.d_block_size = blk;
       cache.d_grid_size  = std::max(prop.multiProcessorCount * hints.grid_factor_hint, 1);
       cache.d_shared_mem = sh;
     }
   }
   cache.initialized = true;
 }
 
 __global__ void kernel_ttmc3_static_block_per_i(
   const uint64_t* __restrict__ mode_0_idx,
   const uint64_t* __restrict__ mode_1_ptr, const uint64_t* __restrict__ mode_1_idx,
   const uint64_t* __restrict__ mode_2_ptr, const uint64_t* __restrict__ mode_2_idx,
   const scalar_t* __restrict__ values, scalar_t* arr_A,  scalar_t* arr_B,  scalar_t* arr_O,
   uint32_t f1, uint32_t f2,  int num_warps)
 {
   extern __shared__ scalar_t buf[];
   __shared__ int s_counter;
   int buf_index;
 
   uint64_t i_ptr = blockIdx.x;
   uint64_t i =  mode_0_idx[i_ptr];
 
   uint32_t warp_size = 32;
   uint32_t warp_id = threadIdx.x / warp_size;
   int tid_in_warp = threadIdx.x % warp_size;
 
   for(int buf_idx = threadIdx.x; buf_idx < f1 * f2; buf_idx += blockDim.x){
     buf[num_warps * f2 + buf_idx] = 0.0;
   }
   if (threadIdx.x == 0) s_counter = 0;
   __syncthreads();
 
   uint64_t offset, j_ptr, j_ptr_offset = mode_1_ptr[i_ptr];
   unsigned int full_mask = 0xFFFFFFFFu;
 
   while(true){
     if(tid_in_warp == 0) offset = atomicAdd(&s_counter, 1);
     offset = __shfl_sync(full_mask, offset, 0);
     j_ptr = j_ptr_offset + offset;
     if(j_ptr < mode_1_ptr[i_ptr + 1]){
       uint64_t j = mode_1_idx[j_ptr];
 
       for(int buf_idx_offset = warp_id * f2; buf_idx_offset < (warp_id + 1)* f2; buf_idx_offset += warp_size){
         buf_index = buf_idx_offset + tid_in_warp;
         if(buf_index < (warp_id + 1)* f2){
           buf[buf_index] = 0.0;
         }
       }
 
       for(uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr){
         uint64_t k = mode_2_idx[k_ptr];
 
         for(uint32_t s_offset = 0; s_offset < f2; s_offset += warp_size){
           uint32_t s = s_offset + tid_in_warp;
           if(s < f2){
             buf[warp_id * f2 + s] += values[k_ptr] * arr_B[k * f2 + s];
           }
         }
       }
 
       for(uint32_t r = 0; r < f1; ++r){
         for(uint32_t s_offset = 0; s_offset < f2; s_offset += warp_size){
           uint32_t s = s_offset + tid_in_warp;
           if(s < f2){
             atomicAdd(&buf[num_warps * f2 + r * f2 + s], buf[warp_id * f2 + s] * arr_A[j * f1 + r]);
           }
         }
       }
     }
     else {
       break;
     }
   }
   __syncthreads();
 
   for(uint32_t r_offset = 0; r_offset < f1; r_offset += num_warps){
     uint32_t r = r_offset + warp_id;
     if(r < f1){
       for(uint32_t s_offset = 0; s_offset < f2; s_offset += warp_size){
         uint32_t s = s_offset + tid_in_warp;
         if(s < f2){
           arr_O[i * f1* f2 + r * f2 + s] += buf[num_warps * f2 + r * f2 + s];
         }
       }
     }
   }
 }
 
 __global__ void kernel_ttmc3_dynamic_tasks(
   const uint64_t* __restrict__ mode_0_idx,
   const uint64_t* __restrict__ mode_1_ptr, const uint64_t* __restrict__ mode_1_idx,
   const uint64_t* __restrict__ mode_2_ptr, const uint64_t* __restrict__ mode_2_idx,
   const scalar_t* __restrict__ values, const scalar_t* __restrict__ arr_A, const scalar_t* __restrict__ arr_B,
   scalar_t* __restrict__ arr_O, uint32_t f1, uint32_t f2,
   const TaskRange* __restrict__ tasks, uint64_t num_tasks,
   unsigned long long* __restrict__ global_task_counter)
 {
   extern __shared__ scalar_t shared_buf[];
   const uint32_t warp_size = 32;
   const uint32_t warp_id = threadIdx.x / warp_size;
   const uint32_t lane = threadIdx.x % warp_size;
   const uint32_t warps_per_block = blockDim.x / warp_size;
 
   const bool use_register_accum = (f1 <= 64) && (f2 <= warp_size);
   scalar_t* warp_temp = nullptr;
   scalar_t* warp_accum = nullptr;
   if (!use_register_accum) {
     warp_temp = shared_buf + warp_id * f2;
     warp_accum = shared_buf + warps_per_block * f2 + warp_id * (size_t)f1 * f2;
   }
 
   const unsigned full_mask = 0xFFFFFFFFu;
   const uint32_t rs = f1 * f2;
 
   if (use_register_accum) {
     while (true) { //each warp picks a task from the task queue dynamically and processes it
       unsigned long long task_idx;
       if (lane == 0) {
         task_idx = atomicAdd(global_task_counter, 1ULL);
       }
       task_idx = __shfl_sync(full_mask, task_idx, 0);
       if (task_idx >= num_tasks) {
         break;
       }
       
       TaskRange task = tasks[task_idx];
       uint64_t i_ptr = task.i_ptr;
       uint64_t j_begin = task.j_begin;
       uint64_t j_end = task.j_end;
       uint64_t k_tile_begin = task.k_begin;
       uint64_t k_tile_end = task.k_end;
 
       uint64_t i = mode_0_idx[i_ptr];
       scalar_t* out_base = arr_O + (i * (uint64_t)f1 * f2);
 
       scalar_t accum_local[64]; //allocated in global memory?
       if (lane < f2) {
 #pragma unroll
         for (uint32_t r = 0; r < f1; ++r) {
           accum_local[r] = 0.0f;
         }
       }
 
       if (k_tile_begin < k_tile_end) {
         uint64_t j_ptr = j_begin;
         uint64_t j = mode_1_idx[j_ptr];
         const scalar_t* __restrict__ a_row = arr_A + (j * (uint64_t)f1);
         scalar_t temp_reg = 0.0f;
         if (lane < f2) {
           temp_reg = 0.0f;
         }
 
         for (uint64_t k_ptr = k_tile_begin; k_ptr < k_tile_end; ++k_ptr) {
           uint64_t k = mode_2_idx[k_ptr];
           scalar_t val = values[k_ptr];
           const scalar_t* __restrict__ b_row = arr_B + (k * (uint64_t)f2);
           if (lane < f2) {
             scalar_t b_val = b_row[lane];
             temp_reg = fma(val, b_val, temp_reg);
           }
         }
 
 #pragma unroll
         for (uint32_t r = 0; r < f1; ++r) {
           scalar_t a = 0.0f;
           if (lane == 0) {
             a = a_row[r];
           }
           a = __shfl_sync(full_mask, a, 0);
           if (lane < f2) {
             accum_local[r] = fma(temp_reg, a, accum_local[r]);
           }
         }
       }
       else {
         for (uint64_t j_ptr = j_begin; j_ptr < j_end; ++j_ptr) {
           uint64_t j = mode_1_idx[j_ptr];
           const scalar_t* __restrict__ a_row = arr_A + (j * (uint64_t)f1);
           scalar_t temp_reg = 0.0f;
           if (lane < f2) {
             temp_reg = 0.0f;
           }
 
           for (uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
             uint64_t k = mode_2_idx[k_ptr];
             scalar_t val = values[k_ptr];
             const scalar_t* __restrict__ b_row = arr_B + (k * (uint64_t)f2);
             if (lane < f2) {
               scalar_t b_val = b_row[lane];
               temp_reg = fma(val, b_val, temp_reg);
             }
           }
 
 #pragma unroll
           for (uint32_t r = 0; r < f1; ++r) {
             scalar_t a = 0.0f;
             if (lane == 0) {
               a = a_row[r];
             }
             a = __shfl_sync(full_mask, a, 0);
             if (lane < f2) {
               accum_local[r] = fma(temp_reg, a, accum_local[r]);
             }
           }
         }
       }
 
       if (lane < f2) {
         for (uint32_t r = 0; r < f1; ++r) {
           atomicAdd(out_base + r * f2 + lane, accum_local[r]);
         }
       }
     }
   }
   else {
     while (true) {
       unsigned long long task_idx;
       if (lane == 0) {
         task_idx = atomicAdd(global_task_counter, 1ULL);
       }
       task_idx = __shfl_sync(full_mask, task_idx, 0);
       if (task_idx >= num_tasks) {
         break;
       }
 
       TaskRange task = tasks[task_idx];
       uint64_t i_ptr = task.i_ptr;
       uint64_t j_begin = task.j_begin;
       uint64_t j_end = task.j_end;
       uint64_t k_tile_begin = task.k_begin;
       uint64_t k_tile_end = task.k_end;
 
       uint64_t i = mode_0_idx[i_ptr];
       scalar_t* out_base = arr_O + (i * (uint64_t)f1 * f2);
 
       for (uint32_t idx = lane; idx < rs; idx += warp_size) {
         warp_accum[idx] = 0.0f;
       }
       __syncwarp(full_mask);
 
       if (k_tile_begin < k_tile_end) {
         uint64_t j_ptr = j_begin;
         uint64_t j = mode_1_idx[j_ptr];
 
         for (uint32_t s = lane; s < f2; s += warp_size) {
           warp_temp[s] = 0.0f;
         }
         __syncwarp(full_mask);
 
         for (uint64_t k_ptr = k_tile_begin; k_ptr < k_tile_end; ++k_ptr) {
           uint64_t k = mode_2_idx[k_ptr];
           scalar_t val = values[k_ptr];
           const scalar_t* __restrict__ b_row = arr_B + (k * (uint64_t)f2);
           for (uint32_t s = lane; s < f2; s += warp_size) {
             scalar_t b_val = b_row[s];
             warp_temp[s] = fma(val, b_val, warp_temp[s]);
           }
         }
         __syncwarp(full_mask);
 
 #pragma unroll
         for (uint32_t r = 0; r < f1; ++r) {
           scalar_t a = 0.0f;
           if (lane == 0) {
             const scalar_t* __restrict__ a_row = arr_A + (j * (uint64_t)f1);
             a = a_row[r];
           }
           a = __shfl_sync(full_mask, a, 0);
           uint32_t base = r * f2;
           for (uint32_t s = lane; s < f2; s += warp_size) {
             warp_accum[base + s] = fma(warp_temp[s], a, warp_accum[base + s]);
           }
         }
         __syncwarp(full_mask);
       }
       else {
         for (uint64_t j_ptr = j_begin; j_ptr < j_end; ++j_ptr) {
           uint64_t j = mode_1_idx[j_ptr];
 
           for (uint32_t s = lane; s < f2; s += warp_size) {
             warp_temp[s] = 0.0f;
           }
           __syncwarp(full_mask);
 
           for (uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
             uint64_t k = mode_2_idx[k_ptr];
             scalar_t val = values[k_ptr];
             const scalar_t* __restrict__ b_row = arr_B + (k * (uint64_t)f2);
             for (uint32_t s = lane; s < f2; s += warp_size) {
               scalar_t b_val = b_row[s];
               warp_temp[s] = fma(val, b_val, warp_temp[s]);
             }
           }
           __syncwarp(full_mask);
 
 #pragma unroll
           for (uint32_t r = 0; r < f1; ++r) {
             scalar_t a = 0.0f;
             if (lane == 0) {
               const scalar_t* __restrict__ a_row = arr_A + (j * (uint64_t)f1);
               a = a_row[r];
             }
             a = __shfl_sync(full_mask, a, 0);
             uint32_t base = r * f2;
             for (uint32_t s = lane; s < f2; s += warp_size) {
               warp_accum[base + s] = fma(warp_temp[s], a, warp_accum[base + s]);
             }
           }
           __syncwarp(full_mask);
         }
       }
 
       for (uint32_t idx = lane; idx < rs; idx += warp_size) {
         atomicAdd(out_base + idx, warp_accum[idx]);
       }
       __syncwarp(full_mask);
     }
   }
 }
 
 __device__ inline void process_segment_4d(
   uint64_t j_ptr,
   uint64_t k_begin,
   uint64_t k_end,
   const uint64_t* __restrict__ mode_1_idx,
   const uint64_t* __restrict__ mode_2_ptr,
   const uint64_t* __restrict__ mode_2_idx,
   const uint64_t* __restrict__ mode_3_ptr,
   const uint64_t* __restrict__ mode_3_idx,
   const scalar_t* __restrict__ values,
   const scalar_t* __restrict__ arr_A,
   const scalar_t* __restrict__ arr_B,
   const scalar_t* __restrict__ arr_C,
   scalar_t* __restrict__ out_base,
   scalar_t* __restrict__ warp_temp,
   scalar_t* __restrict__ warp_b,
   uint32_t f1, uint32_t f2, uint32_t f3,
   uint32_t lane, unsigned mask)
 {
   const uint32_t warp_size = 32;
   const uint32_t rs = f2 * f3;
 
   for (uint32_t idx = lane; idx < rs; idx += warp_size) {
     warp_temp[idx] = 0.0f;
   }
   __syncwarp(mask);
 
   if (k_begin < k_end) {
     for (uint64_t k_ptr = k_begin; k_ptr < k_end; ++k_ptr) {
       uint64_t k = mode_2_idx[k_ptr];
       const scalar_t* __restrict__ b_row = arr_B + (k * (uint64_t)f2);
       for (uint32_t r2 = lane; r2 < f2; r2 += warp_size) {
         warp_b[r2] = b_row[r2];
       }
       __syncwarp(mask);
 
       uint64_t l_begin = mode_3_ptr[k_ptr];
       uint64_t l_end = mode_3_ptr[k_ptr + 1];
 
       if (f3 <= warp_size) {
         scalar_t tc = 0.0f;
         for (uint64_t l_ptr = l_begin; l_ptr < l_end; ++l_ptr) {
           uint64_t l = mode_3_idx[l_ptr];
           scalar_t val = values[l_ptr];
           if (lane < f3) {
             tc = fma(val, arr_C[l * (uint64_t)f3 + lane], tc);
           }
         }
         for (uint32_t r2 = 0; r2 < f2; ++r2) {
           if (lane < f3) {
             warp_temp[r2 * f3 + lane] = fma(warp_b[r2], tc, warp_temp[r2 * f3 + lane]);
           }
         }
       } else {
         for (uint64_t l_ptr = l_begin; l_ptr < l_end; ++l_ptr) {
           uint64_t l = mode_3_idx[l_ptr];
           scalar_t val = values[l_ptr];
           const scalar_t* __restrict__ c_row = arr_C + (l * (uint64_t)f3);
           for (uint32_t idx = lane; idx < rs; idx += warp_size) {
             uint32_t r2 = idx / f3;
             uint32_t r3 = idx % f3;
             scalar_t scaled_b = warp_b[r2] * val;
             warp_temp[idx] = fma(scaled_b, c_row[r3], warp_temp[idx]);
           }
           __syncwarp(mask);
         }
       }
     }
   }
 
   uint64_t j = mode_1_idx[j_ptr];
   const scalar_t* __restrict__ a_row = arr_A + (j * (uint64_t)f1);
   for (uint32_t r1 = 0; r1 < f1; ++r1) {
     scalar_t a = (lane == 0) ? a_row[r1] : 0.0f;
     a = __shfl_sync(mask, a, 0);
     if (a == 0.0f) {
       continue;
     }
     uint64_t base = (uint64_t)r1 * rs;
     for (uint32_t idx = lane; idx < rs; idx += warp_size) {
       atomicAdd(out_base + base + idx, warp_temp[idx] * a);
     }
   }
   __syncwarp(mask);
 }
 
 __global__ void kernel_ttmc4_dynamic_tasks(
   const uint64_t* __restrict__ mode_0_idx,
   const uint64_t* __restrict__ mode_1_ptr, const uint64_t* __restrict__ mode_1_idx,
   const uint64_t* __restrict__ mode_2_ptr, const uint64_t* __restrict__ mode_2_idx,
   const uint64_t* __restrict__ mode_3_ptr, const uint64_t* __restrict__ mode_3_idx,
   const scalar_t* __restrict__ values,
   const scalar_t* __restrict__ arr_A, const scalar_t* __restrict__ arr_B, const scalar_t* __restrict__ arr_C,
   scalar_t* __restrict__ arr_O,
   uint32_t f1, uint32_t f2, uint32_t f3,
   const TaskRange* __restrict__ tasks, uint64_t num_tasks,
   unsigned long long* __restrict__ global_task_counter)
 {
   extern __shared__ scalar_t shared_buf[];
   const uint32_t warp_size = 32;
   const uint32_t warp_id = threadIdx.x / warp_size;
   const uint32_t lane = threadIdx.x % warp_size;
   const uint32_t warps_per_block = blockDim.x / warp_size;
   const unsigned mask = 0xFFFFFFFFu;
   const size_t rs = (size_t)f2 * f3;
 
   scalar_t* warp_temp = shared_buf + warp_id * rs;
   scalar_t* warp_b = shared_buf + (size_t)warps_per_block * rs + warp_id * (size_t)f2;
 
   while (true) {
     unsigned long long task_idx;
     if (lane == 0) {
       task_idx = atomicAdd(global_task_counter, 1ULL);
     }
     task_idx = __shfl_sync(mask, task_idx, 0);
     if (task_idx >= num_tasks) {
       break;
     }
 
     TaskRange task = tasks[task_idx];
     uint64_t i_ptr = task.i_ptr;
     uint64_t i = mode_0_idx[i_ptr];
     scalar_t* out_base = arr_O + (i * (uint64_t)f1 * (uint64_t)rs);
 
     if (task.k_begin < task.k_end) {
       uint64_t j_ptr = task.j_begin;
       process_segment_4d(
         j_ptr, task.k_begin, task.k_end,
         mode_1_idx, mode_2_ptr, mode_2_idx, mode_3_ptr, mode_3_idx,
         values, arr_A, arr_B, arr_C,
         out_base, warp_temp, warp_b,
         f1, f2, f3, lane, mask
       );
     }
     else {
       uint64_t j_begin = task.j_begin;
       uint64_t j_end = task.j_end;
       for (uint64_t j_ptr = j_begin; j_ptr < j_end; ++j_ptr) {
         uint64_t k_begin = mode_2_ptr[j_ptr];
         uint64_t k_end = mode_2_ptr[j_ptr + 1];
         process_segment_4d(
           j_ptr, k_begin, k_end,
           mode_1_idx, mode_2_ptr, mode_2_idx, mode_3_ptr, mode_3_idx,
           values, arr_A, arr_B, arr_C,
           out_base, warp_temp, warp_b,
           f1, f2, f3, lane, mask
         );
       }
     }
   }
 }
 
 __global__ void kernel_ttmc4_tiny_stream(
   const uint64_t* __restrict__ mode_1_idx,
   const uint64_t* __restrict__ mode_2_ptr, const uint64_t* __restrict__ mode_2_idx,
   const uint64_t* __restrict__ mode_3_ptr, const uint64_t* __restrict__ mode_3_idx,
   const scalar_t* __restrict__ values,
   const scalar_t* __restrict__ factor_A,
   const scalar_t* __restrict__ factor_B,
   const scalar_t* __restrict__ factor_C,
   scalar_t* arr_O,
   uint32_t f1, uint32_t f2, uint32_t f3,
   uint64_t j_ptr_offset, uint64_t i)
{
  extern __shared__ scalar_t buf[];
  uint64_t j_ptr = j_ptr_offset + blockIdx.x;
  uint64_t j = mode_1_idx[j_ptr];

  int buf_ofst = f2 * f3;

  for (int buf_index = threadIdx.y * blockDim.x + threadIdx.x;
       buf_index < buf_ofst; buf_index += blockDim.x * blockDim.y) {
    buf[buf_index] = 0.0f;
  }
  __syncthreads();

  for (uint64_t k_ptr = mode_2_ptr[j_ptr];
       k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
    uint64_t k = mode_2_idx[k_ptr];

    int buf_index = threadIdx.y * blockDim.x + threadIdx.x;
    if (buf_index < (int)f3) {
      buf[buf_ofst + buf_index] = 0.0f;
    }
    __syncthreads();

    for (uint64_t l_ptr_ofst = mode_3_ptr[k_ptr];
         l_ptr_ofst < mode_3_ptr[k_ptr + 1];
         l_ptr_ofst += blockDim.y) {
      uint64_t l_ptr = l_ptr_ofst + threadIdx.y;
      if (l_ptr < mode_3_ptr[k_ptr + 1]) {
        uint64_t l = mode_3_idx[l_ptr];
        for (uint32_t t_ofst = 0; t_ofst < f3; t_ofst += blockDim.x) {
          uint32_t t = t_ofst + threadIdx.x;
          if (t < f3) {
            atomicAdd(&buf[buf_ofst + t], values[l_ptr] *
              factor_C[l * f3 + t]);
          }
        }
      }
    }
    __syncthreads();

    for (uint32_t s_ofst = 0; s_ofst < f2; s_ofst += blockDim.y) {
      uint32_t s = s_ofst + threadIdx.y;
      if (s < f2) {
        for (uint32_t t_ofst = 0; t_ofst < f3; t_ofst += blockDim.x) {
          uint32_t t = t_ofst + threadIdx.x;
          if (t < f3) {
            atomicAdd(&buf[s * f3 + t], buf[buf_ofst + t] *
              factor_B[k * f2 + s]);
          }
        }
      }
    }
    __syncthreads();
  }
  __syncthreads();

  for (uint32_t r = 0; r < f1; ++r) {
    for (uint32_t s_ofst = 0; s_ofst < f2; s_ofst += blockDim.y) {
      uint32_t s = s_ofst + threadIdx.y;
      if (s < f2) {
        for (uint32_t t_ofst = 0; t_ofst < f3; t_ofst += blockDim.x) {
          uint32_t t = t_ofst + threadIdx.x;
          if (t < f3) {
            atomicAdd(&arr_O[ i * f1 * f2 * f3
              + r * f2 * f3
              + s * f3
              + t],
              buf[s * f3 + t] * factor_A[j * f1 + r]);
          }
        }
      }
    }
  }
}
 
static double launch_ttmc4_tiny_streams_flat(
  uint64_t** h_mode_ptrs, uint64_t** h_mode_idxs,   // HOST: for launch loop
  uint64_t** d_mode_ptrs, uint64_t** d_mode_idxs,   // GPU: for kernel args
  const scalar_t* d_values,
  const scalar_t* d_factor_A,
  const scalar_t* d_factor_B,
  const scalar_t* d_factor_C,
  scalar_t* d_arr_O,
  uint32_t f1, uint32_t f2, uint32_t f3,
  uint64_t num_i,
  int stream_hint)
{
  if (num_i == 0) {
    return 0.0;
  }

  int default_streams = std::max(1, stream_hint);
  uint64_t desired_streams = static_cast<uint64_t>(default_streams);
  if (desired_streams > num_i) {
    desired_streams = num_i;
  }
  std::vector<cudaStream_t> streams(desired_streams);
  for (uint64_t s = 0; s < desired_streams; ++s) {
    cudaCheckError(cudaStreamCreate(&streams[s]));
  }

  dim3 blockDim(32, 32);
  size_t sharedMemBytes = (size_t)f2 * f3 * sizeof(scalar_t) + (size_t)f3 * sizeof(scalar_t);

  auto start = std::chrono::high_resolution_clock::now();
  for (uint64_t i_ptr = 0; i_ptr < num_i; ++i_ptr) {
    uint64_t begin = h_mode_ptrs[1][i_ptr];
    uint64_t end = h_mode_ptrs[1][i_ptr + 1];
    if (begin >= end) continue;
    dim3 gridDim(static_cast<unsigned int>(end - begin));
    if (gridDim.x == 0) continue;
    uint64_t i = h_mode_idxs[0][i_ptr];
    cudaStream_t stream = streams[i_ptr % desired_streams];
    kernel_ttmc4_tiny_stream<<<gridDim, blockDim, sharedMemBytes, stream>>>(
      d_mode_idxs[1],
      d_mode_ptrs[2], d_mode_idxs[2],
      d_mode_ptrs[3], d_mode_idxs[3],
      d_values,
      d_factor_A, d_factor_B, d_factor_C,
      d_arr_O, f1, f2, f3,
      begin, i
    );
  }
  cudaCheckError(cudaGetLastError());
  for (uint64_t s = 0; s < desired_streams; ++s) {
    cudaCheckError(cudaStreamSynchronize(streams[s]));
    cudaCheckError(cudaStreamDestroy(streams[s]));
  }
  cudaCheckError(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  return static_cast<double>(duration) / 1000.0;
}
 
 
__global__ void compute_inv_sigma(const scalar_t* W, scalar_t* diag, 
  int N, int R, scalar_t eps) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < R)
    diag[j] = (scalar_t)1 / sqrt(fmax(W[N - R + j], eps));
}

// Check resolution condition for top-R eigenvalues of Gram matrix.
// Eigenvalues in W are ascending: W[0] <= ... <= W[dim-1].
// Top-R are W[dim-R] ... W[dim-1].
// GNR_{j,j+1} = (W[j+1] - W[j]) / (2 * dim * eps_mach * W[dim-1])
// If any consecutive pair has GNR <= 1, sets flag[0] = 1.
__global__ void check_resolution(const scalar_t* W, int c, int dim, int R,
  scalar_t eps_mach, int* flag, scalar_t trace) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < R - 1) {
    scalar_t lam_j  = W[dim - R + j];
    scalar_t lam_j1 = W[dim - R + j + 1];
    scalar_t gnr = (lam_j1 - lam_j) / ((scalar_t)2 * c * eps_mach * trace);
    if (gnr <= (scalar_t)1)
      atomicExch(flag, 1);
  }
}

// ---- Double-precision kernels for Gram EVD path ----

// Double-precision resolution check (uses DBL_EPSILON always)
__global__ void check_resolution_dp(const double* W, int c, int dim, int R,
  double eps_mach_dp, int* flag, double trace) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < R - 1) {
    double lam_j  = W[dim - R + j];
    double lam_j1 = W[dim - R + j + 1];
    double gnr = (lam_j1 - lam_j) / (2.0 * c * eps_mach_dp * trace);
    if (gnr <= 1.0)
      atomicExch(flag, 1);
  }
}

// Compute 1/sigma from double eigenvalues, output to scalar_t
__global__ void compute_inv_sigma_dp(const double* W, scalar_t* diag,
  int N, int R, double eps) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < R)
    diag[j] = (scalar_t)(1.0 / sqrt(fmax(W[N - R + j], eps)));
}


// Convert double array to scalar_t (no-op when scalar_t is double)
__global__ void convert_dp_to_scalar(const double* src, scalar_t* dst, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = (scalar_t)src[i];
}

// Convert scalar_t array to double
__global__ void convert_scalar_to_dp(const scalar_t* src, double* dst, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = (double)src[i];
}

 // All tensor and factor data must already be on GPU before calling.
 // Result is written to d_arr_O (GPU) and stays there — caller copies to HOST as needed.
void run_ttmc_cuda(
  uint64_t** d_mode_ptrs,    // GPU: CSFCopy.d_ptrs.data() — pre-uploaded ptr arrays
  uint64_t** d_mode_idxs,    // GPU: CSFCopy.d_idxs.data() — pre-uploaded idx arrays
  scalar_t*  d_values,       // GPU: CSFCopy.d_values — pre-uploaded values
  scalar_t** d_factor_mats,  // GPU: CSF-level indexed factor matrices (null at ncm level)
  scalar_t*  d_arr_O,        // GPU: pre-allocated output buffer (zeroed here)
  uint64_t   arr_O_size,
  int ncm,
  uint64_t* ranks, int order,
  TTMcCache& cache,
  bool log_method = false)
 {
   std::vector<int> other_modes;
   other_modes.reserve(order - 1);
   for (int mode = 0; mode < order; ++mode)
     if (mode != ncm) other_modes.push_back(mode);
   if (other_modes.size() < 2)
     throw std::runtime_error("Tensor order too small for contraction");
   int idx_A = other_modes[0];
   int idx_B = other_modes[1];
   int idx_C = (other_modes.size() > 2) ? other_modes[2] : -1;
   int f1 = (int)ranks[idx_A];
   int f2 = (int)ranks[idx_B];
   int f3 = (idx_C >= 0) ? (int)ranks[idx_C] : 0;
 
   cudaCheckError(cudaMemset(d_arr_O, 0, sizeof(scalar_t) * arr_O_size));
 
   switch (ncm) {
     case 0: {
       if (order == 3) {
         if (cache.prefer_static) {
           auto start = std::chrono::high_resolution_clock::now();
           kernel_ttmc3_static_block_per_i<<<cache.s_grid_size, cache.s_block_size, cache.s_shared_mem>>>(
             d_mode_idxs[0],
             d_mode_ptrs[1], d_mode_idxs[1],
             d_mode_ptrs[2], d_mode_idxs[2],
             d_values, d_factor_mats[idx_A], d_factor_mats[idx_B], d_arr_O,
             f1, f2, cache.s_num_warps);
           cudaCheckError(cudaGetLastError());
           cudaCheckError(cudaDeviceSynchronize());
           auto end = std::chrono::high_resolution_clock::now();
           if (log_method) {
             cout << "Method: kernel_ttmc3_static_block_per_i, Time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / 1000.0
                  << " ms" << endl;
           }
         } else {
           cudaCheckError(cudaMemset(cache.d_task_counter, 0, sizeof(unsigned long long)));
           cudaCheckError(cudaFuncSetAttribute(kernel_ttmc3_dynamic_tasks,
             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cache.d_shared_mem));
           auto start = std::chrono::high_resolution_clock::now();
           if (!cache.host_tasks.empty()) {
             kernel_ttmc3_dynamic_tasks<<<cache.d_grid_size, cache.d_block_size, cache.d_shared_mem>>>(
               d_mode_idxs[0],
               d_mode_ptrs[1], d_mode_idxs[1],
               d_mode_ptrs[2], d_mode_idxs[2],
               d_values, d_factor_mats[idx_A], d_factor_mats[idx_B], d_arr_O, f1, f2,
               cache.d_tasks, (uint64_t)cache.host_tasks.size(), cache.d_task_counter);
           }
           cudaCheckError(cudaGetLastError());
           cudaCheckError(cudaDeviceSynchronize());
           auto end = std::chrono::high_resolution_clock::now();
           if (log_method) {
             cout << "Method: kernel_ttmc3_dynamic_tasks, Time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / 1000.0
                  << " ms" << endl;
           }
         }
       }
       else if (order == 4) {
         if (cache.prefer_tiny_streams) {
           // tiny_streams needs HOST mode data for its launch loop; use ptrs cached at prepare time.
           uint64_t num_i = cache.size_mode_ptr[1] - 1;
           double duration_ms = launch_ttmc4_tiny_streams_flat(
             cache.h_mode_ptrs.data(), cache.h_mode_idxs.data(),
             d_mode_ptrs, d_mode_idxs,
             d_values,
             d_factor_mats[idx_A], d_factor_mats[idx_B], d_factor_mats[idx_C],
             d_arr_O,
             f1, f2, f3,
             num_i,
             cache.tiny_stream_count);
           if (log_method) {
             cout << "Method: launch_ttmc4_tiny_streams_flat/kernel_ttmc4_tiny_stream, Time: "
                  << duration_ms << " ms" << endl;
           }
         } else {
           cudaCheckError(cudaMemset(cache.d_task_counter, 0, sizeof(unsigned long long)));
           cudaCheckError(cudaFuncSetAttribute(kernel_ttmc4_dynamic_tasks,
             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cache.d_shared_mem));
           auto start = std::chrono::high_resolution_clock::now();
           if (!cache.host_tasks.empty()) {
             kernel_ttmc4_dynamic_tasks<<<cache.d_grid_size, cache.d_block_size, cache.d_shared_mem>>>(
               d_mode_idxs[0],
               d_mode_ptrs[1], d_mode_idxs[1],
               d_mode_ptrs[2], d_mode_idxs[2],
               d_mode_ptrs[3], d_mode_idxs[3],
               d_values,
               d_factor_mats[idx_A], d_factor_mats[idx_B], d_factor_mats[idx_C],
               d_arr_O,
               f1, f2, f3,
               cache.d_tasks, (uint64_t)cache.host_tasks.size(), cache.d_task_counter);
           }
           cudaCheckError(cudaGetLastError());
           cudaCheckError(cudaDeviceSynchronize());
           auto end = std::chrono::high_resolution_clock::now();
           if (log_method) {
             cout << "Method: kernel_ttmc4_dynamic_tasks, Time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / 1000.0
                  << " ms" << endl;
           }
         }
       }
       break;
     }
     default:
       throw std::runtime_error("Unsupported ncm in run_ttmc_cuda. Only ncm=0 is implemented.");
   }
   cudaCheckError(cudaDeviceSynchronize());
   // Result is in d_arr_O on GPU. No copies or frees here — all owned by caller.
 }
 // ===================================================================
 // (End of v4-Optimized TTMc Engine)
 // ===================================================================
 
 
 // ===================================================================
 // TTMc GPU Kernel (ncm_0): one block per root fiber.
 // Computes arr_O[i * f1*f2 + r*f2 + s] += sum_j sum_k X[i,j,k]*A[j,r]*B[k,s]
 // i = mode_0_idx[blockIdx.x]  (non-contracting / root mode index)
 // arr_A = factor for level-1 (middle) mode, arr_B = factor for level-2 (leaf) mode.
 // Both factors are row-major on GPU: A[row,col] = arr_A[row*f1 + col].
 // ===================================================================
 __global__ void GPU_4L_CM_device_func_ncm_0(
   const uint64_t* __restrict__ mode_0_idx,
   const uint64_t* __restrict__ mode_1_ptr, const uint64_t* __restrict__ mode_1_idx,
   const uint64_t* __restrict__ mode_2_ptr, const uint64_t* __restrict__ mode_2_idx,
   const scalar_t* __restrict__ values, scalar_t* arr_A,  scalar_t* arr_B,  scalar_t* arr_O,
   uint32_t f1, uint32_t f2,  int num_warps)
 {
   // Renamed from 'buf' to 'hooi_sbuf' to avoid extern __shared__ name collision
   // with the float buf[] declared in the copied v4 kernels above.
   extern __shared__ scalar_t hooi_sbuf[];
   __shared__ int s_counter;
   int buf_index;
 
   uint64_t i_ptr = blockIdx.x;
   uint64_t i =  mode_0_idx[i_ptr];
 
   uint32_t warp_size = 32;
   uint32_t warp_id = threadIdx.x / warp_size;
   int tid_in_warp = threadIdx.x % warp_size;
 
   for(int buf_idx = threadIdx.x; buf_idx < (int)(f1 * f2); buf_idx += blockDim.x)
     hooi_sbuf[num_warps * f2 + buf_idx] = (scalar_t)0;
   if (threadIdx.x == 0) s_counter = 0;
   __syncthreads();
 
   uint64_t offset, j_ptr, j_ptr_offset = mode_1_ptr[i_ptr];
   unsigned int full_mask = 0xFFFFFFFFu;
 
   while(true){
     if(tid_in_warp == 0) offset = atomicAdd(&s_counter, 1);
     offset = __shfl_sync(full_mask, offset, 0);
     j_ptr = j_ptr_offset + offset;
     if(j_ptr < mode_1_ptr[i_ptr + 1]){
       uint64_t j = mode_1_idx[j_ptr];
 
       for(int buf_idx_offset = warp_id * f2; buf_idx_offset < (int)((warp_id + 1)* f2); buf_idx_offset += warp_size){
         buf_index = buf_idx_offset + tid_in_warp;
         if(buf_index < (int)((warp_id + 1)* f2))
           hooi_sbuf[buf_index] = (scalar_t)0;
       }
 
       for(uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr){
         uint64_t k = mode_2_idx[k_ptr];
         for(uint32_t s_offset = 0; s_offset < f2; s_offset += warp_size){
           uint32_t s = s_offset + tid_in_warp;
           if(s < f2)
             hooi_sbuf[warp_id * f2 + s] += values[k_ptr] * arr_B[k * f2 + s];
         }
       }
 
       for(uint32_t r = 0; r < f1; ++r){
         for(uint32_t s_offset = 0; s_offset < f2; s_offset += warp_size){
           uint32_t s = s_offset + tid_in_warp;
           if(s < f2)
             atomicAdd(&hooi_sbuf[num_warps * f2 + r * f2 + s], hooi_sbuf[warp_id * f2 + s] * arr_A[j * f1 + r]);
         }
       }
     } else {
       break;
     }
   }
   __syncthreads();
 
   for(uint32_t r_offset = 0; r_offset < f1; r_offset += num_warps){
     uint32_t r = r_offset + warp_id;
     if(r < f1){
       for(uint32_t s_offset = 0; s_offset < f2; s_offset += warp_size){
         uint32_t s = s_offset + tid_in_warp;
         if(s < f2)
           arr_O[i * f1* f2 + r * f2 + s] += hooi_sbuf[num_warps * f2 + r * f2 + s];
       }
     }
   }
 }
 
 
 #define CHECK_CUDA(call)                                                        \
   do {                                                                        \
     cudaError_t err = (call);                                               \
     if (err != cudaSuccess) {                                               \
       std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
            << " \"" << cudaGetErrorString(err) << "\"\n";            \
       std::exit(EXIT_FAILURE);                                            \
     }                                                                       \
   } while (0)
 
 inline const char* cusolverStatusString(cusolverStatus_t s) {
   switch (s) {
     case CUSOLVER_STATUS_SUCCESS:                return "SUCCESS";
     case CUSOLVER_STATUS_NOT_INITIALIZED:        return "NOT_INITIALIZED";
     case CUSOLVER_STATUS_ALLOC_FAILED:           return "ALLOC_FAILED";
     case CUSOLVER_STATUS_INVALID_VALUE:          return "INVALID_VALUE";
     case CUSOLVER_STATUS_ARCH_MISMATCH:          return "ARCH_MISMATCH";
     case CUSOLVER_STATUS_MAPPING_ERROR:          return "MAPPING_ERROR";
     case CUSOLVER_STATUS_EXECUTION_FAILED:       return "EXECUTION_FAILED";
     case CUSOLVER_STATUS_INTERNAL_ERROR:         return "INTERNAL_ERROR";
     case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "MATRIX_TYPE_NOT_SUPPORTED";
     case CUSOLVER_STATUS_NOT_SUPPORTED:          return "NOT_SUPPORTED";
     case CUSOLVER_STATUS_ZERO_PIVOT:             return "ZERO_PIVOT";
     case CUSOLVER_STATUS_INVALID_LICENSE:        return "INVALID_LICENSE";
     default:                                     return "UNKNOWN";
   }
 }
 
 #define CHECK_CUSOLVER(call)                                                    \
   do {                                                                        \
     cusolverStatus_t status = (call);                                       \
     if (status != CUSOLVER_STATUS_SUCCESS) {                                \
       std::cerr << "cuSOLVER error at " << __FILE__ << ":" << __LINE__    \
            << " status=" << cusolverStatusString(status) << "\n";     \
       std::exit(EXIT_FAILURE);                                            \
     }                                                                       \
   } while (0)
 
#define CHECK_CUBLAS(call)                                                      \
  do {                                                                        \
    cublasStatus_t status = (call);                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                  \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__      \
           << " status=" << static_cast<int>(status) << "\n";        \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
  } while (0)

static std::string formatBytes(double bytes);

static bool topREigenvaluesPassResolutionHost(
  const std::vector<double>& top_eigs,
  int c,
  double trace)
{
  if (top_eigs.size() < 2) return true;
  const double eps_mach_dp = 1.1102230246251565e-16;
  for (size_t j = 0; j + 1 < top_eigs.size(); ++j) {
    double gnr = (top_eigs[j + 1] - top_eigs[j]) / (2.0 * c * eps_mach_dp * trace);
    if (gnr <= 1.0) return false;
  }
  return true;
}

static bool solveTopREigensystemDpExact(
  cusolverDnHandle_t cusolverH,
  double* d_gram_dp,
  int dim,
  int rank,
  int c,
  double trace,
  scalar_t** d_VR_out,
  scalar_t** d_inv_sigma_out,
  double* core_norm_sq_out,
  bool* resolution_failed_out,
  bool verbose)
{
  if (resolution_failed_out) *resolution_failed_out = false;
  if (d_VR_out) *d_VR_out = nullptr;
  if (d_inv_sigma_out) *d_inv_sigma_out = nullptr;
  if (core_norm_sq_out) *core_norm_sq_out = 0.0;
  rank = std::min(rank, dim);
  if (rank <= 0) return false;

  double* d_W_dp = nullptr;
  void* d_work_dp = nullptr;
  void* h_work = nullptr;
  int* d_info = nullptr;
  size_t workspace_bytes_device = 0;
  size_t workspace_bytes_host = 0;
  int64_t meig = 0;
  const int64_t il = dim - rank + 1;
  const int64_t iu = dim;
  cusolverDnParams_t params = nullptr;

  CHECK_CUDA(cudaMalloc(&d_W_dp, sizeof(double) * dim));

  CHECK_CUSOLVER(cusolverDnCreateParams(&params));

  cusolverStatus_t st = cusolverDnXsyevdx_bufferSize(
    cusolverH, params,
    CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I,
    CUBLAS_FILL_MODE_UPPER,
    (int64_t)dim,
    CUDA_R_64F, d_gram_dp, (int64_t)dim,
    nullptr, nullptr, il, iu, &meig,
    CUDA_R_64F, d_W_dp,
    CUDA_R_64F,
    &workspace_bytes_device,
    &workspace_bytes_host);
  if (st != CUSOLVER_STATUS_SUCCESS) {
    if (verbose) {
      std::cout << "  exact top-R eigensolver bufferSize failed with status="
                << cusolverStatusString(st) << "\n";
    }
    CHECK_CUDA(cudaFree(d_W_dp));
    cusolverDnDestroyParams(params);
    return false;
  }

  if (verbose) {
    size_t free_bytes = 0, total_bytes = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::cout << "  exact top-R eigensolver workspace: device "
              << formatBytes((double)workspace_bytes_device)
              << " host " << formatBytes((double)workspace_bytes_host)
              << "  free GPU before work alloc " << formatBytes((double)free_bytes)
              << "\n";
  }

  cudaError_t err = cudaMalloc(&d_work_dp, std::max<size_t>(workspace_bytes_device, 1));
  if (err != cudaSuccess) {
    cudaGetLastError();
    if (verbose) std::cout << "  exact top-R eigensolver work alloc failed\n";
    CHECK_CUDA(cudaFree(d_W_dp));
    cusolverDnDestroyParams(params);
    return false;
  }
  if (workspace_bytes_host > 0) {
    h_work = std::malloc(workspace_bytes_host);
    if (!h_work) {
      if (verbose) std::cout << "  exact top-R eigensolver host work alloc failed\n";
      CHECK_CUDA(cudaFree(d_W_dp));
      CHECK_CUDA(cudaFree(d_work_dp));
      cusolverDnDestroyParams(params);
      return false;
    }
  }
  CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

  st = cusolverDnXsyevdx(
    cusolverH, params,
    CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I,
    CUBLAS_FILL_MODE_UPPER,
    (int64_t)dim,
    CUDA_R_64F, d_gram_dp, (int64_t)dim,
    nullptr, nullptr, il, iu, &meig,
    CUDA_R_64F, d_W_dp,
    CUDA_R_64F,
    d_work_dp, workspace_bytes_device,
    h_work, workspace_bytes_host,
    d_info);
  if (st != CUSOLVER_STATUS_SUCCESS) {
    if (verbose) {
      std::cout << "  exact top-R eigensolver failed with status="
                << cusolverStatusString(st) << "\n";
    }
    CHECK_CUDA(cudaFree(d_W_dp));
    CHECK_CUDA(cudaFree(d_work_dp));
    CHECK_CUDA(cudaFree(d_info));
    if (h_work) std::free(h_work);
    cusolverDnDestroyParams(params);
    return false;
  }

  int h_info = 0;
  CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info != 0 || meig < rank) {
    if (verbose) {
      std::cout << "  exact top-R eigensolver returned info=" << h_info
                << " meig=" << meig << "\n";
    }
    CHECK_CUDA(cudaFree(d_W_dp));
    CHECK_CUDA(cudaFree(d_work_dp));
    CHECK_CUDA(cudaFree(d_info));
    if (h_work) std::free(h_work);
    cusolverDnDestroyParams(params);
    return false;
  }

  std::vector<double> h_top(rank);
  CHECK_CUDA(cudaMemcpy(
    h_top.data(),
    d_W_dp,
    sizeof(double) * rank,
    cudaMemcpyDeviceToHost));
  if (!topREigenvaluesPassResolutionHost(h_top, c, trace)) {
    if (resolution_failed_out) *resolution_failed_out = true;
    if (verbose) {
      std::cout << "  exact top-R eigensolver resolution check failed among top-" << rank
                << " eigenvalues\n";
    }
    CHECK_CUDA(cudaFree(d_W_dp));
    CHECK_CUDA(cudaFree(d_work_dp));
    CHECK_CUDA(cudaFree(d_info));
    if (h_work) std::free(h_work);
    cusolverDnDestroyParams(params);
    return false;
  }

  if (core_norm_sq_out) {
    double sum = 0.0;
    for (double lam : h_top) sum += std::max(lam, 0.0);
    *core_norm_sq_out = sum;
  }

  std::vector<scalar_t> h_inv_sigma(rank);
  for (int j = 0; j < rank; ++j) {
    h_inv_sigma[j] = (scalar_t)(1.0 / std::sqrt(std::max(h_top[j], 1e-12)));
  }

  CHECK_CUDA(cudaMalloc(d_VR_out, sizeof(scalar_t) * (long long)dim * rank));
  convert_dp_to_scalar<<<((long long)dim * rank + 255) / 256, 256>>>(
    d_gram_dp, *d_VR_out, (long long)dim * rank);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMalloc(d_inv_sigma_out, sizeof(scalar_t) * rank));
  CHECK_CUDA(cudaMemcpy(
    *d_inv_sigma_out,
    h_inv_sigma.data(),
    sizeof(scalar_t) * rank,
    cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaFree(d_W_dp));
  CHECK_CUDA(cudaFree(d_work_dp));
  CHECK_CUDA(cudaFree(d_info));
  if (h_work) std::free(h_work);
  cusolverDnDestroyParams(params);
  return true;
}

static bool symmetricEigenvaluesGpuSmall(
  cusolverDnHandle_t cusolverH,
  double* d_a,
  int n,
  std::vector<double>& evals_out,
  bool verbose)
{
  evals_out.assign(n, 0.0);
  double* d_w = nullptr;
  double* d_work = nullptr;
  int* d_info = nullptr;
  int lwork = 0;

  CHECK_CUDA(cudaMalloc(&d_w, sizeof(double) * n));
  cusolverStatus_t st = cusolverDnDsyevd_bufferSize(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
    n, d_a, n, d_w, &lwork);
  if (st != CUSOLVER_STATUS_SUCCESS) {
    if (verbose) {
      std::cout << "  small symmetric eig workspace query failed status="
                << cusolverStatusString(st) << "\n";
    }
    CHECK_CUDA(cudaFree(d_w));
    return false;
  }

  CHECK_CUDA(cudaMalloc(&d_work, sizeof(double) * std::max(lwork, 1)));
  CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
  st = cusolverDnDsyevd(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
    n, d_a, n, d_w, d_work, lwork, d_info);
  if (st != CUSOLVER_STATUS_SUCCESS) {
    if (verbose) {
      std::cout << "  small symmetric eig failed status="
                << cusolverStatusString(st) << "\n";
    }
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(d_info));
    return false;
  }

  int h_info = 0;
  CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info != 0) {
    if (verbose) {
      std::cout << "  small symmetric eig failed info=" << h_info << "\n";
    }
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(d_info));
    return false;
  }

  CHECK_CUDA(cudaMemcpy(evals_out.data(), d_w, sizeof(double) * n, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_w));
  CHECK_CUDA(cudaFree(d_work));
  CHECK_CUDA(cudaFree(d_info));
  return true;
}

__global__ static void extractTrailingColumnsKernel(
  const double* src,
  double* dst,
  int rows,
  int cols,
  int keep_cols)
{
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long)rows * keep_cols;
  if (idx >= total) return;
  int row = idx % rows;
  int col = idx / rows;
  int src_col = cols - keep_cols + col;
  dst[idx] = src[(long long)row + (long long)src_col * rows];
}

static bool orthonormalizeBasisGpu(
  cusolverDnHandle_t cusolverH,
  cublasHandle_t cublasH,
  const double* d_in,
  int n,
  int k,
  double* d_out,
  bool verbose)
{
  double* d_s = nullptr;
  double* d_inv = nullptr;
  CHECK_CUDA(cudaMalloc(&d_s, sizeof(double) * k * k));
  const double alpha = 1.0;
  const double beta = 0.0;
  CHECK_CUBLAS(cublasDsyrk(
    cublasH, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
    k, n, &alpha, d_in, n, &beta, d_s, k));

  std::vector<double> h_eval;
  if (!symmetricEigenvaluesGpuSmall(cusolverH, d_s, k, h_eval, verbose)) {
    CHECK_CUDA(cudaFree(d_s));
    return false;
  }
  std::vector<double> h_inv(k);
  for (int j = 0; j < k; ++j) {
    if (h_eval[j] <= 1e-20) {
      if (verbose) std::cout << "  orthonormalization failed: nonpositive Gram eigenvalue\n";
      CHECK_CUDA(cudaFree(d_s));
      return false;
    }
    h_inv[j] = 1.0 / std::sqrt(h_eval[j]);
  }
  CHECK_CUDA(cudaMalloc(&d_inv, sizeof(double) * k));
  CHECK_CUDA(cudaMemcpy(d_inv, h_inv.data(), sizeof(double) * k, cudaMemcpyHostToDevice));

  CHECK_CUBLAS(cublasDgemm(
    cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
    n, k, k, &alpha, d_in, n, d_s, k, &beta, d_out, n));
  CHECK_CUBLAS(cublasDdgmm(
    cublasH, CUBLAS_SIDE_RIGHT,
    n, k, d_out, n, d_inv, 1, d_out, n));

  CHECK_CUDA(cudaFree(d_s));
  CHECK_CUDA(cudaFree(d_inv));
  return true;
}

static bool solveTopREigensystemDpIterativeGpu(
  cusolverDnHandle_t cusolverH,
  cublasHandle_t cublasH,
  double* d_gram_dp,
  int dim,
  int rank,
  int c,
  double trace,
  scalar_t** d_VR_out,
  scalar_t** d_inv_sigma_out,
  double* core_norm_sq_out,
  bool* resolution_failed_out,
  bool verbose)
{
  if (resolution_failed_out) *resolution_failed_out = false;
  if (d_VR_out) *d_VR_out = nullptr;
  if (d_inv_sigma_out) *d_inv_sigma_out = nullptr;
  if (core_norm_sq_out) *core_norm_sq_out = 0.0;
  rank = std::min(rank, dim);
  if (rank <= 0) return false;

  int oversample = std::max(0, getEnvInt("TTMC_GPU_ITER_OVERSAMPLE", 8));
  int max_iters = std::max(1, getEnvInt("TTMC_GPU_ITER_ITERS", 8));
  int k = std::min(dim, rank + oversample);
  if (verbose) {
    std::cout << "  trying iterative GPU top-R eigensolver: block=" << k
              << " iters=" << max_iters << "\n";
  }

  double* d_q = nullptr;
  double* d_z = nullptr;
  CHECK_CUDA(cudaMalloc(&d_q, sizeof(double) * (size_t)dim * k));
  CHECK_CUDA(cudaMalloc(&d_z, sizeof(double) * (size_t)dim * k));

  std::vector<double> h_init((size_t)dim * k);
  std::mt19937_64 rng(1234567);
  std::normal_distribution<double> normal(0.0, 1.0);
  for (double& x : h_init) x = normal(rng);
  CHECK_CUDA(cudaMemcpy(d_q, h_init.data(), sizeof(double) * h_init.size(), cudaMemcpyHostToDevice));
  if (!orthonormalizeBasisGpu(cusolverH, cublasH, d_q, dim, k, d_z, verbose)) {
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_z));
    return false;
  }
  std::swap(d_q, d_z);

  const double alpha = 1.0;
  const double beta = 0.0;
  for (int it = 0; it < max_iters; ++it) {
    CHECK_CUBLAS(cublasDsymm(
      cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
      dim, k, &alpha, d_gram_dp, dim, d_q, dim, &beta, d_z, dim));
    if (!orthonormalizeBasisGpu(cusolverH, cublasH, d_z, dim, k, d_q, verbose)) {
      CHECK_CUDA(cudaFree(d_q));
      CHECK_CUDA(cudaFree(d_z));
      return false;
    }
  }

  CHECK_CUBLAS(cublasDsymm(
    cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
    dim, k, &alpha, d_gram_dp, dim, d_q, dim, &beta, d_z, dim));

  double* d_t = nullptr;
  CHECK_CUDA(cudaMalloc(&d_t, sizeof(double) * k * k));
  CHECK_CUBLAS(cublasDgemm(
    cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
    k, k, dim, &alpha, d_q, dim, d_z, dim, &beta, d_t, k));

  std::vector<double> h_eval;
  if (!symmetricEigenvaluesGpuSmall(cusolverH, d_t, k, h_eval, verbose)) {
    CHECK_CUDA(cudaFree(d_t));
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_z));
    return false;
  }

  std::vector<double> h_top(rank);
  for (int j = 0; j < rank; ++j) h_top[j] = h_eval[k - rank + j];
  if (!topREigenvaluesPassResolutionHost(h_top, c, trace)) {
    if (resolution_failed_out) *resolution_failed_out = true;
    if (verbose) {
      std::cout << "  iterative GPU eigensolver resolution check failed among top-" << rank
                << " eigenvalues\n";
    }
    CHECK_CUDA(cudaFree(d_t));
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_z));
    return false;
  }

  std::vector<scalar_t> h_inv_sigma(rank);
  double core_sum = 0.0;
  for (int j = 0; j < rank; ++j) {
    core_sum += std::max(h_top[j], 0.0);
    h_inv_sigma[j] = (scalar_t)(1.0 / std::sqrt(std::max(h_top[j], 1e-12)));
  }
  if (core_norm_sq_out) *core_norm_sq_out = core_sum;

  double* d_vselect = nullptr;
  CHECK_CUDA(cudaMalloc(&d_vselect, sizeof(double) * (size_t)k * rank));
  extractTrailingColumnsKernel<<<((long long)k * rank + 255) / 256, 256>>>(d_t, d_vselect, k, k, rank);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUBLAS(cublasDgemm(
    cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
    dim, rank, k, &alpha, d_q, dim, d_vselect, k, &beta, d_z, dim));

  CHECK_CUDA(cudaMalloc(d_VR_out, sizeof(scalar_t) * (size_t)dim * rank));
  convert_dp_to_scalar<<<((long long)dim * rank + 255) / 256, 256>>>(
    d_z, *d_VR_out, (long long)dim * rank);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMalloc(d_inv_sigma_out, sizeof(scalar_t) * rank));
  CHECK_CUDA(cudaMemcpy(
    *d_inv_sigma_out,
    h_inv_sigma.data(),
    sizeof(scalar_t) * rank,
    cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaFree(d_vselect));
  CHECK_CUDA(cudaFree(d_t));
  CHECK_CUDA(cudaFree(d_q));
  CHECK_CUDA(cudaFree(d_z));
  if (verbose) {
    std::cout << "  iterative GPU top-R eigensolver completed\n";
  }
  return true;
}

// ===================================================================
// GPU memory tracking (enabled by -g / --gpu-stats flag)
 // Shadows cudaMalloc so every allocation prints its size when the flag is set.
 // Must be defined after cuda_runtime.h is included.
 // ===================================================================
 static bool g_gpu_stats = false;
 uint64_t total_gpu_malloc_size = 0;
 inline cudaError_t _tracked_cudaMalloc(void** ptr, size_t bytes,
                                        const char* file, int line) {
   if (g_gpu_stats) {
     double mb = bytes / (1024.0 * 1024.0);
     double gb = mb / 1024.0;
     std::cout << "[GPU malloc] "
               << mb << " MB  (" << gb << " GB)  " << file << ":" << line << "\n";
     total_gpu_malloc_size += bytes;
     std::cout << "[Total GPU malloc] " << total_gpu_malloc_size / (1024.0 * 1024.0 * 1024.0) << " GB\n";
   }
   return cudaMalloc(ptr, bytes);
 }
 #define cudaMalloc(ptr, size) _tracked_cudaMalloc((void**)(ptr), (size), __FILE__, __LINE__)
 
 
 // Initialize factor matrix with random orthonormal columns (Gram-Schmidt)
 void init_factor_orthonormal(uint64_t rows, uint64_t cols, unsigned int seed, scalar_t* A) {
   std::mt19937 gen(seed);
   std::normal_distribution<scalar_t> dist((scalar_t)0, (scalar_t)1);
   for (uint64_t i = 0; i < rows * cols; i++) A[i] = dist(gen);
   for (uint64_t c = 0; c < cols; c++) {
     scalar_t norm = 0;
     for (uint64_t r = 0; r < rows; r++) norm += A[r * cols + c] * A[r * cols + c];
     norm = std::sqrt(norm);
     if (norm < (scalar_t)1e-10) norm = (scalar_t)1;
     for (uint64_t r = 0; r < rows; r++) A[r * cols + c] /= norm;
     for (uint64_t c2 = c + 1; c2 < cols; c2++) {
       scalar_t dot = 0;
       for (uint64_t r = 0; r < rows; r++) dot += A[r * cols + c] * A[r * cols + c2];
       for (uint64_t r = 0; r < rows; r++) A[r * cols + c2] -= dot * A[r * cols + c];
     }
   }
 }

static std::string factorBinPath(const std::string& dir, int mode)
{
  return dir + "/mode" + std::to_string(mode) + ".bin";
}

struct FactorFilePayload
{
  uint64_t rows{0};
  uint64_t cols{0};
  std::vector<float> values;
};

static FactorFilePayload read_factor_file_float32(
  const std::string& path,
  uint64_t expected_rows,
  uint64_t max_cols)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open factor file: " + path);
  }

  in.seekg(0, std::ios::end);
  const std::streamsize file_size = in.tellg();
  in.seekg(0, std::ios::beg);
  if (file_size < 0) {
    throw std::runtime_error("failed to stat factor file: " + path);
  }

  const std::streamsize bytes_per_row =
    static_cast<std::streamsize>(expected_rows * sizeof(float));

  auto read_values = [&](uint64_t rows, uint64_t cols, std::streamsize offset) {
    FactorFilePayload payload;
    payload.rows = rows;
    payload.cols = cols;
    payload.values.resize(static_cast<size_t>(rows * cols));
    in.seekg(offset, std::ios::beg);
    in.read(reinterpret_cast<char*>(payload.values.data()),
            static_cast<std::streamsize>(payload.values.size() * sizeof(float)));
    if (!in) {
      throw std::runtime_error("failed to read factor file: " + path);
    }
    return payload;
  };

  if (file_size >= static_cast<std::streamsize>(2 * sizeof(uint64_t))) {
    uint64_t fr = 0, fc = 0;
    in.read(reinterpret_cast<char*>(&fr), sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(&fc), sizeof(uint64_t));
    if (!in) {
      throw std::runtime_error("failed to read factor header: " + path);
    }
    const std::streamsize header_bytes =
      static_cast<std::streamsize>(2 * sizeof(uint64_t) + fr * fc * sizeof(float));
    if (fr == expected_rows &&
        fc > 0 &&
        fc <= max_cols &&
        header_bytes == file_size) {
      return read_values(fr, fc, static_cast<std::streamsize>(2 * sizeof(uint64_t)));
    }
    in.clear();
  }

  if (bytes_per_row <= 0 || file_size % bytes_per_row != 0) {
    std::ostringstream oss;
    oss << "factor file size mismatch for " << path
        << ": cannot infer raw float32 layout for " << expected_rows << " rows";
    throw std::runtime_error(oss.str());
  }

  const uint64_t inferred_cols =
    static_cast<uint64_t>(file_size / bytes_per_row);
  if (inferred_cols == 0 || inferred_cols > max_cols) {
    std::ostringstream oss;
    oss << "factor file column mismatch for " << path
        << ": inferred " << inferred_cols
        << " cols for " << expected_rows << " rows, max allowed " << max_cols;
    throw std::runtime_error(oss.str());
  }
  return read_values(expected_rows, inferred_cols, 0);
}

static void dump_factor_bin(const std::string& path, const scalar_t* A, uint64_t rows, uint64_t cols)
{
  std::filesystem::path out_path(path);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
   }
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open factor dump path: " + path);
  }
  std::vector<float> buf(static_cast<size_t>(rows * cols));
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<float>(A[i]);
  }
  out.write(reinterpret_cast<const char*>(&rows), sizeof(uint64_t));
  out.write(reinterpret_cast<const char*>(&cols), sizeof(uint64_t));
  out.write(reinterpret_cast<const char*>(buf.data()),
            static_cast<std::streamsize>(buf.size() * sizeof(float)));
  if (!out) {
    throw std::runtime_error("failed to write factor dump path: " + path);
  }
}

static bool load_factor_bin(const std::string& path, scalar_t* A, uint64_t rows, uint64_t cols)
{
  std::ifstream probe(path, std::ios::binary);
  if (!probe) return false;
  probe.close();

  FactorFilePayload payload = read_factor_file_float32(path, rows, cols);
  if (payload.rows != rows || payload.cols != cols) {
    std::ostringstream oss;
    oss << "factor shape mismatch for " << path
        << ": file " << payload.rows << "x" << payload.cols
        << ", expected " << rows << "x" << cols;
    throw std::runtime_error(oss.str());
  }
  for (size_t i = 0; i < payload.values.size(); ++i) {
    A[i] = static_cast<scalar_t>(payload.values[i]);
  }
  return true;
}
 
 // Function for aligned memory allocation
 scalar_t* allocate_aligned_array(size_t num_elements) {
   constexpr size_t alignment = 32;                    // 32 bytes = 256 bits
   constexpr size_t element_size = sizeof(scalar_t);
 
   size_t total_bytes = num_elements * element_size;
 
   // Pad to next multiple of 32 bytes if needed
   if (total_bytes % alignment != 0) {
     total_bytes = ((total_bytes + alignment - 1) / alignment) * alignment;
   }
 
   // Now, allocate aligned memory
   void* ptr = std::aligned_alloc(alignment, total_bytes);
   if (!ptr) {
     throw std::runtime_error("Failed to allocate aligned memory");
   }
 
   //initilaize to zero
   size_t total_elements = total_bytes / element_size;
   scalar_t* arr = static_cast<scalar_t*>(ptr);
   for (size_t i = 0; i < total_elements; ++i) {
     arr[i] = (scalar_t)0;
   }
 
   return static_cast<scalar_t*>(ptr);
 }
 
 // Frobenius norm squared (sum of v^2 over all nnz values)
 scalar_t frobenius_norm_sq_sparse(const scalar_t* values, size_t nnz) {
   double sum = 0;
   for (size_t i = 0; i < nnz; i++) { double v = values[i]; sum += v * v; }
   return static_cast<scalar_t>(sum);
 }


  // Full SVD: cusolverDnSgesvd when M > N (typical); eigendecomposition of AA^T
 // when M <= N (degenerate small mode — cuSOLVER's Sgesvd doesn't support M<N reliably).
 // d_A: M×N col-major (destroyed on return). Output: top-R left singular vectors
 // in d_factor (M×R col-major).
void gpu_full_svd_update_factor(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH,
  scalar_t* d_A, int M, int N, int R, scalar_t* d_factor, bool verbose) {
  int min_mn = std::min(M, N);
  R = std::min(R, min_mn);

  cudaEvent_t ev0, ev1; float ev_ms = 0.f;
  cudaEventCreate(&ev0); cudaEventCreate(&ev1);
  if (verbose) std::cout << "  M = " << M << ", N = " << N << ", R = " << R << "\n";
  
  if (M > N) {
    // --- cusolverDnSgesvd / cusolverDnDgesvd: jobu='S', jobvt='N' (works when M > N) ---
    scalar_t *d_S, *d_U, *d_VT_dummy; int *d_info;
    CHECK_CUDA(cudaMalloc(&d_S,        sizeof(scalar_t) * min_mn));
    CHECK_CUDA(cudaMalloc(&d_U,        sizeof(scalar_t) * M * min_mn));
    CHECK_CUDA(cudaMalloc(&d_VT_dummy, sizeof(scalar_t)));
    CHECK_CUDA(cudaMalloc(&d_info,     sizeof(int)));
    // d_A is row-major (M,N) = col-major B=A^T (N,M) lda=N.
    // gesvd needs col-major A (M×N). Transpose B->A on GPU.
    scalar_t *d_colmaj_A;
    CHECK_CUDA(cudaMalloc(&d_colmaj_A, sizeof(scalar_t) * M * N));
    { scalar_t one=(scalar_t)1, zero=(scalar_t)0;
      CHECK_CUBLAS(cublasGeamT(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, M, N,
        &one, d_A, N, &zero, d_colmaj_A, M, d_colmaj_A, M)); }

    int lwork = 0;
    CHECK_CUSOLVER(cusolverGesvdBufSizeT(cusolverH, M, N, &lwork));
    scalar_t *d_work;
    // std::cout << "ratio of d_work to MxN matrix size= " << (double) sizeof(scalar_t) * std::max(lwork, 1) / (M * N * sizeof(scalar_t)) << "\n";
    CHECK_CUDA(cudaMalloc(&d_work, sizeof(scalar_t) * std::max(lwork, 1)));


    // cublasMath_t mode;
    // cublasGetMathMode(cublasH, &mode);
    // if (verbose) std::cout << "Math mode: " << mode << "\n";
    // 0 = CUBLAS_DEFAULT_MATH (TF32 on Ampere+)
    // 1 = CUBLAS_PEDANTIC_MATH (strict FP32)

    cudaEventRecord(ev0);
    CHECK_CUSOLVER(cusolverGesvdT(cusolverH, 'S', 'N', M, N, d_colmaj_A, M,
      d_S, d_U, M, d_VT_dummy, 1, d_work, lwork, nullptr, d_info));
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&ev_ms, ev0, ev1);
    if (verbose) std::cout << "  cusolverDnGesvd: " << ev_ms << " ms\n";

    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0)
      std::cerr << "  WARNING: cusolverDnGesvd info=" << h_info << "\n";

    // Transpose first R cols of d_U (col-major M*R lda=M) -> col-major (R,M) lda=R = row-major (M,R).
    // C[r,m] = d_U[m,r] = u_r[m]
    { scalar_t one=(scalar_t)1, zero=(scalar_t)0;
      CHECK_CUBLAS(cublasGeamT(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R, M,
        &one, d_U, M, &zero, d_factor, R, d_factor, R)); }
    CHECK_CUDA(cudaFree(d_colmaj_A));
    CHECK_CUDA(cudaFree(d_S)); CHECK_CUDA(cudaFree(d_U));
    CHECK_CUDA(cudaFree(d_VT_dummy)); CHECK_CUDA(cudaFree(d_info));
    CHECK_CUDA(cudaFree(d_work));
  } else {
    // --- M <= N: SVD of A^T (N×M, now tall-skinny N > M) ---
    // d_A is row-major (M,N), i.e. col-major A^T (N,M) with lda=N.
    // A^T = P·S·VT  =>  A = VT^T·S·P^T  =>  left sing vecs of A = cols of VT^T = rows of VT.
    // Use jobvt='S' to get VT (M×M col-major), then extract: d_factor = VT^T[:, 0:R].
    // M is tiny (< N = rank product) so all allocations here are negligible.
    scalar_t one = (scalar_t)1, zero = (scalar_t)0;

    // d_A is already A^T: col-major (N, M) with lda=N (N >= M since M <= N).
    // gesvd overwrites its input, so factorize a copy to keep d_A unchanged
    // (the caller still uses d_A for core-norm computation after SVD).
    scalar_t *d_A_tall, *d_S, *d_P, *d_VT; int *d_info;
    CHECK_CUDA(cudaMalloc(&d_A_tall, sizeof(scalar_t) * (size_t)N * (size_t)M));
    CHECK_CUDA(cudaMemcpy(d_A_tall, d_A, sizeof(scalar_t) * (size_t)N * (size_t)M,
      cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMalloc(&d_S,    sizeof(scalar_t) * M));        // min(N,M)=M singular values
    CHECK_CUDA(cudaMalloc(&d_P,    sizeof(scalar_t) * N * M));    // jobu='S': left vecs of A^T (N×M)
    CHECK_CUDA(cudaMalloc(&d_VT,   sizeof(scalar_t) * M * M));    // jobvt='S': right vecs of A^T (M×M) = U^T of A
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
    int lwork = 0;
    CHECK_CUSOLVER(cusolverGesvdBufSizeT(cusolverH, N, M, &lwork));
    scalar_t *d_work;
    CHECK_CUDA(cudaMalloc(&d_work, sizeof(scalar_t) * std::max(lwork, 1)));
    if (verbose) std::cout << "  d_work (MB): " << sizeof(scalar_t) * std::max(lwork, 1) / (1024.0 * 1024.0) << "\n";
    cudaEventRecord(ev0);
    CHECK_CUSOLVER(cusolverGesvdT(cusolverH, 'S', 'S', N, M, d_A_tall, N,
      d_S, d_P, N, d_VT, M, d_work, lwork, nullptr, d_info));
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&ev_ms, ev0, ev1);
    if (verbose) std::cout << "  cusolverDnGesvd(A^T): " << ev_ms << " ms\n";

    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0)
      std::cerr << "  WARNING: cusolverDnGesvd(A^T) info=" << h_info << "\n";

    // First R rows of VT repacked as col-major (R,M) lda=R = row-major (M,R).
    // C[r,m] = d_VT[r + m*M] = VT[r,m] = u_r[m]
    CHECK_CUBLAS(cublasGeamT(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, R, M,
      &one, d_VT, M, &zero, d_factor, R, d_factor, R));

    CHECK_CUDA(cudaFree(d_A_tall));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_P));  CHECK_CUDA(cudaFree(d_VT));
    CHECK_CUDA(cudaFree(d_info)); CHECK_CUDA(cudaFree(d_work));
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEventDestroy(ev0); cudaEventDestroy(ev1);
}

 // Truncated SVD via eigendecomposition of the Gram matrix.
// d_A: row-major (M, N) — equivalently, col-major A^T of shape (N, M) with lda=N.
// Output: top-R left singular vectors in d_factor (M×R col-major).
// verbose: if false, suppress all internal timing prints.
void gpu_truncated_svd_update_factor(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH,
  scalar_t* d_A, int M, int N, int R, scalar_t* d_factor, bool verbose,
  bool allow_iterative_after_exact = true) {
#if !SCALAR_DOUBLE
   (void)allow_iterative_after_exact;
#endif
   scalar_t alpha = (scalar_t)1, beta = (scalar_t)0;
   int K = std::min(M, N);
   R = std::min(R, K);
 
   cudaEvent_t ev_start, ev_stop;
   float ev_ms = 0.f;
   cudaEventCreate(&ev_start);
   cudaEventCreate(&ev_stop);
 
// =====================================================================
// When scalar_t is already double, use direct GEMM/EVD — no casting.
// When scalar_t is float, cast to double for Gram + EVD precision.
// =====================================================================
#if SCALAR_DOUBLE
// #if temp
   // --- scalar_t == double: direct path (no type conversion needed) ---
   if (M > N) {
     bool use_syrk = useSyrkForGram(N);
     scalar_t* d_Gram;
     cudaEventRecord(ev_start);
     CHECK_CUDA(cudaMalloc(&d_Gram, sizeof(scalar_t) * N * N));
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  ATA alloc: " << ev_ms << " ms\n";

     cudaEventRecord(ev_start);
     if (use_syrk) {
       CHECK_CUBLAS(cublasSyrkT(cublasH, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
         N, M, &alpha, d_A, N, &beta, d_Gram, N));
     } else {
       CHECK_CUBLAS(cublasGemmT(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
         N, N, M, &alpha, d_A, N, d_A, N, &beta, d_Gram, N));
     }
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  ATA " << (use_syrk ? "syrk" : "gemm") << ": " << ev_ms << " ms\n";

     // trace(A^T A) = ||A||_F^2  — dasum on diagonal (all entries >= 0 for Gram)
     scalar_t traceATA;
     CHECK_CUBLAS(cublasDasumT(cublasH, N, d_Gram, N + 1, &traceATA));
     if (verbose) std::cout << "  trace(ATA): " << traceATA << "\n";

     scalar_t *d_W;
     CHECK_CUDA(cudaMalloc(&d_W, sizeof(scalar_t) * N));
     int lwork = 0;
     CHECK_CUSOLVER(cusolverSyevdBufSizeT(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
       CUBLAS_FILL_MODE_UPPER, N, d_Gram, N, d_W, &lwork));

     scalar_t *d_work; int *d_info;
     cudaEventRecord(ev_start);
     CHECK_CUDA(cudaMalloc(&d_work, sizeof(scalar_t) * lwork));
     CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  eig buf alloc: " << ev_ms << " ms\n";
     if (verbose) std::cout << "  d_work (MB): " << sizeof(scalar_t) * lwork / (1024.0 * 1024.0) << "\n";

     cudaEventRecord(ev_start);
     CHECK_CUSOLVER(cusolverSyevdT(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
       CUBLAS_FILL_MODE_UPPER, N, d_Gram, N, d_W, d_work, lwork, d_info));
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  eig decomp: " << ev_ms << " ms\n";
     int h_info = 0;
     CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
     if (h_info != 0) {
       std::cerr << "  WARNING: cusolverSyevdT info=" << h_info << "\n";
       CHECK_CUDA(cudaFree(d_Gram)); CHECK_CUDA(cudaFree(d_W));
       CHECK_CUDA(cudaFree(d_work)); CHECK_CUDA(cudaFree(d_info));
       return;
     }

     CHECK_CUDA(cudaMemset(d_info, 0, sizeof(int)));
     if (R > 1)
      check_resolution<<<1, R - 1>>>(d_W, M + N, N, R, eps_mach, d_info, traceATA);
     CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
     if (h_info != 0) {
       if (verbose)
         std::cout << "  resolution condition failed among top-" << R
                   << " eigenvalues, falling back to full SVD\n";
       CHECK_CUDA(cudaFree(d_Gram)); CHECK_CUDA(cudaFree(d_W));
       CHECK_CUDA(cudaFree(d_work)); CHECK_CUDA(cudaFree(d_info));
       gpu_full_svd_update_factor_pinned_workspace(cusolverH, cublasH, d_A, M, N, R, d_factor, plan, ws, verbose);
       return;
     }

     scalar_t* d_V_R = d_Gram + (long long)(N - R) * N;
     cudaEventRecord(ev_start);
     CHECK_CUBLAS(cublasGemmT(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
       R, M, N, &alpha, d_V_R, N, d_A, N, &beta, d_factor, R));
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  AV gemm: " << ev_ms << " ms\n";

     scalar_t* d_diag;
     CHECK_CUDA(cudaMalloc(&d_diag, sizeof(scalar_t) * R));
     cudaEventRecord(ev_start);
     compute_inv_sigma<<<(R+255)/256, 256>>>(d_W, d_diag, N, R, (scalar_t)1e-12);
     CHECK_CUBLAS(cublasDgmmT(cublasH, CUBLAS_SIDE_LEFT, R, M,
       d_factor, R, d_diag, 1, d_factor, R));
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  normalize (1/sigma): " << ev_ms << " ms\n";

     CHECK_CUDA(cudaFree(d_diag));
     CHECK_CUDA(cudaFree(d_Gram)); CHECK_CUDA(cudaFree(d_W));
     CHECK_CUDA(cudaFree(d_work)); CHECK_CUDA(cudaFree(d_info));
   } else {
    scalar_t* d_Gram;
    cudaEventRecord(ev_start);
    CHECK_CUDA(cudaMalloc(&d_Gram, sizeof(scalar_t) * M * M));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  AA^T alloc: " << ev_ms << " ms\n";

    cudaEventRecord(ev_start);
    CHECK_CUBLAS(cublasSyrkT(cublasH, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
        M, N, &alpha, d_A, N, &beta, d_Gram, M));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  AA^T syrk: " << ev_ms << " ms\n";

    // trace(AA^T) = ||A||_F^2  — dasum on diagonal (all entries >= 0 for Gram)
    scalar_t traceAAT;
    CHECK_CUBLAS(cublasDasumT(cublasH, M, d_Gram, M + 1, &traceAAT));
    if (verbose) std::cout << "  trace(AAT): " << traceAAT << "\n";

    scalar_t *d_W;
    CHECK_CUDA(cudaMalloc(&d_W, sizeof(scalar_t) * M));
    int lwork = 0;
    CHECK_CUSOLVER(cusolverSyevdBufSizeT(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER, M, d_Gram, M, d_W, &lwork));

    scalar_t *d_work; int *d_info;
    cudaEventRecord(ev_start);
    CHECK_CUDA(cudaMalloc(&d_work, sizeof(scalar_t) * lwork));
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  eig buf alloc: " << ev_ms << " ms\n";

    cudaEventRecord(ev_start);
    CHECK_CUSOLVER(cusolverSyevdT(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER, M, d_Gram, M, d_W, d_work, lwork, d_info));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  eig decomp: " << ev_ms << " ms\n";

    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::cerr << "  syevd failed: devInfo = " << h_info << "\n";
    }

    CHECK_CUDA(cudaMemset(d_info, 0, sizeof(int)));
    if (R > 1)
    check_resolution<<<1, R - 1>>>(d_W, N + M, M, R, eps_mach, d_info, traceAAT);
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
      if (verbose)
        std::cout << "  resolution condition failed among top-" << R
                  << " eigenvalues, falling back to full SVD\n";
      CHECK_CUDA(cudaFree(d_Gram)); CHECK_CUDA(cudaFree(d_W));
      CHECK_CUDA(cudaFree(d_work)); CHECK_CUDA(cudaFree(d_info));
      gpu_full_svd_update_factor(cusolverH, cublasH, d_A, M, N, R, d_factor, verbose);
      return;
    }

    cudaEventRecord(ev_start);
    { scalar_t one = (scalar_t)1, zero = (scalar_t)0;
      CHECK_CUBLAS(cublasGeamT(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
          R, M,
          &one,  d_Gram + (long long)(M - R) * M, M,
          &zero, d_factor, R,
          d_factor, R));
    }
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  transpose eigvecs: " << ev_ms << " ms\n";

    CHECK_CUDA(cudaFree(d_Gram)); CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_work)); CHECK_CUDA(cudaFree(d_info));
}
#else
   // --- scalar_t == float: cast to double for Gram + EVD precision ---
   if (M > N) {
     const double eps_mach_dp = 1.1102230246251565e-16; // DBL_EPSILON
     bool use_syrk = useSyrkForGram(N);
     double* d_Gram_dp;
     cudaEventRecord(ev_start);
     CHECK_CUDA(cudaMalloc(&d_Gram_dp, sizeof(double) * N * N));
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  ATA alloc (dp): " << ev_ms << " ms\n";

     // A^T*A in double. Try full M×N double alloc; fall back to blocked if OOM.
     double alpha_dp = 1.0, beta_zero = 0.0;
     long long MN = (long long)M * N;
     double* d_A_dp = nullptr;
     cudaError_t alloc_err = cudaMalloc(&d_A_dp, sizeof(double) * MN);

     bool use_blocked = false;
     cudaEventRecord(ev_start);
     if (alloc_err == cudaSuccess) {
       convert_scalar_to_dp<<<(MN + 255)/256, 256>>>(d_A, d_A_dp, MN);
       CHECK_CUDA(cudaGetLastError());
       cublasStatus_t gram_st = CUBLAS_STATUS_SUCCESS;
       if (use_syrk) {
         gram_st = cublasDsyrk(cublasH, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
           N, M, &alpha_dp, d_A_dp, N, &beta_zero, d_Gram_dp, N);
       } else {
         gram_st = cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
           N, N, M, &alpha_dp, d_A_dp, N, d_A_dp, N, &beta_zero, d_Gram_dp, N);
       }
       CHECK_CUDA(cudaFree(d_A_dp));
       if (gram_st != CUBLAS_STATUS_SUCCESS) {
         if (verbose) std::cout << "  full dp " << (use_syrk ? "syrk" : "gemm")
                                << " failed (status=" << gram_st
                                << "), falling back to blocked\n";
         use_blocked = true;
       }
     } else {
       cudaGetLastError();
       if (verbose) std::cout << "  full dp alloc failed, using blocked path\n";
       use_blocked = true;
     }
     if (use_blocked) {
       int block_rows = std::min(std::max(1, getEnvInt("TTMC_BLOCKED_DP_ROWS", 32768)), M);
       double* d_blk_dp = nullptr;
       while (block_rows > 0) {
         cudaError_t blk_err = cudaMalloc(&d_blk_dp, sizeof(double) * (long long)block_rows * N);
         if (blk_err == cudaSuccess) break;
         cudaGetLastError();
         if (verbose) {
           std::cout << "  blocked dp alloc failed for " << block_rows
                     << " rows, retrying smaller block\n";
         }
         block_rows /= 2;
       }
       if (!d_blk_dp) {
         std::cerr << "  blocked dp alloc failed even after shrinking block rows\n";
         CHECK_CUDA(cudaErrorMemoryAllocation);
       }
       double beta_one = 1.0;
       for (int s = 0; s < M; s += block_rows) {
         int b = std::min(block_rows, M - s);
         long long chunk_elems = (long long)b * N;
         convert_scalar_to_dp<<<(chunk_elems + 255)/256, 256>>>(
           d_A + (long long)s * N, d_blk_dp, chunk_elems);
         if (use_syrk) {
           CHECK_CUBLAS(cublasDsyrk(cublasH, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
             N, b, &alpha_dp, d_blk_dp, N,
             (s == 0) ? &beta_zero : &beta_one, d_Gram_dp, N));
         } else {
           CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
             N, N, b, &alpha_dp, d_blk_dp, N, d_blk_dp, N,
             (s == 0) ? &beta_zero : &beta_one, d_Gram_dp, N));
         }
       }
       CHECK_CUDA(cudaFree(d_blk_dp));
     }
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  ATA " << (use_syrk ? "syrk" : "gemm") << " (dp): " << ev_ms << " ms\n";

     // trace(A^T A) = ||A||_F^2  — dasum on diagonal (all entries >= 0 for Gram)
     double traceATA;
     CHECK_CUBLAS(cublasDasum(cublasH, N, d_Gram_dp, N + 1, &traceATA));
     if (verbose) std::cout << "  trace(ATA): " << traceATA << "\n";

     double *d_W_dp;
     CHECK_CUDA(cudaMalloc(&d_W_dp, sizeof(double) * N));
     int lwork = 0;
     bool used_topr_solver = false;
     scalar_t* d_V_R = nullptr;
     scalar_t* d_diag = nullptr;
     double* d_work_dp = nullptr;
     int* d_info = nullptr;

     cusolverStatus_t full_st = CUSOLVER_STATUS_SUCCESS;
     if (!used_topr_solver) {
       full_st = cusolverDnDsyevd_bufferSize(
         cusolverH, CUSOLVER_EIG_MODE_VECTOR,
         CUBLAS_FILL_MODE_UPPER, N, d_Gram_dp, N, d_W_dp, &lwork);
       if (full_st != CUSOLVER_STATUS_SUCCESS) {
         if (verbose) {
           std::cout << "  full eigensolver bufferSize failed with status="
                     << cusolverStatusString(full_st)
                     << ", deferring to late tiled fallback\n";
          }
          used_topr_solver = true;
       } else {
         cudaEventRecord(ev_start);
         cudaError_t work_err = cudaMalloc(&d_work_dp, sizeof(double) * std::max(lwork, 1));
         if (work_err != cudaSuccess) {
           cudaGetLastError();
           used_topr_solver = true;
           if (verbose) {
             std::cout << "  full eigensolver work alloc failed, deferring to late tiled fallback\n";
           }
         } else {
           CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
           cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
           cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
           if (verbose) std::cout << "  eig buf alloc: " << ev_ms << " ms\n";
           if (verbose) std::cout << "  d_work (MB): " << sizeof(double) * lwork / (1024.0 * 1024.0) << "\n";

           cudaEventRecord(ev_start);
           full_st = cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
             CUBLAS_FILL_MODE_UPPER, N, d_Gram_dp, N, d_W_dp, d_work_dp, lwork, d_info);
           cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
           cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
           if (verbose) std::cout << "  eig decomp (dp): " << ev_ms << " ms\n";
           if (full_st != CUSOLVER_STATUS_SUCCESS) {
             if (verbose) {
               std::cout << "  full eigensolver failed with status="
                         << cusolverStatusString(full_st)
                         << ", deferring to late tiled fallback\n";
             }
             CHECK_CUDA(cudaFree(d_work_dp));
             d_work_dp = nullptr;
             CHECK_CUDA(cudaFree(d_info));
             d_info = nullptr;
             used_topr_solver = true;
           }
         }
       }
     }

     if (used_topr_solver) {
       CHECK_CUDA(cudaFree(d_Gram_dp));
       CHECK_CUDA(cudaFree(d_W_dp));
       throw ExactTopRFullGramFallbackFailed();
     }

     if (!used_topr_solver) {
       int h_info = 0;
       CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
       if (h_info != 0) {
         std::cerr << "  WARNING: cusolverDnDsyevd info=" << h_info << "\n";
         CHECK_CUDA(cudaFree(d_Gram_dp)); CHECK_CUDA(cudaFree(d_W_dp));
         CHECK_CUDA(cudaFree(d_work_dp)); CHECK_CUDA(cudaFree(d_info));
         return;
       }

       CHECK_CUDA(cudaMemset(d_info, 0, sizeof(int)));
       if (R > 1)
        check_resolution_dp<<<1, R - 1>>>(d_W_dp, M + N, N, R, eps_mach_dp, d_info, traceATA);
       CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
       if (h_info != 0) {
         if (verbose)
           std::cout << "  resolution condition failed among top-" << R
                     << " eigenvalues, falling back to full SVD\n";
         CHECK_CUDA(cudaFree(d_Gram_dp)); CHECK_CUDA(cudaFree(d_W_dp));
         CHECK_CUDA(cudaFree(d_work_dp)); CHECK_CUDA(cudaFree(d_info));
         gpu_full_svd_update_factor(cusolverH, cublasH, d_A, M, N, R, d_factor, verbose);
         return;
       }

       // Convert top-R eigenvectors double→float (N×R, small)
       double* d_V_R_dp = d_Gram_dp + (long long)(N - R) * N;
       CHECK_CUDA(cudaMalloc(&d_V_R, sizeof(scalar_t) * N * R));
       convert_dp_to_scalar<<<((long long)N*R + 255)/256, 256>>>(d_V_R_dp, d_V_R, N * R);
       CHECK_CUDA(cudaMalloc(&d_diag, sizeof(scalar_t) * R));
       compute_inv_sigma_dp<<<(R+255)/256, 256>>>(d_W_dp, d_diag, N, R, 1e-12);
     }
     CHECK_CUDA(cudaGetLastError());

     cudaEventRecord(ev_start);
     CHECK_CUBLAS(cublasGemmT(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
       R, M, N, &alpha, d_V_R, N, d_A, N, &beta, d_factor, R));
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  AV gemm: " << ev_ms << " ms\n";

     cudaEventRecord(ev_start);
     CHECK_CUBLAS(cublasDgmmT(cublasH, CUBLAS_SIDE_LEFT, R, M,
       d_factor, R, d_diag, 1, d_factor, R));
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  normalize (1/sigma): " << ev_ms << " ms\n";

     CHECK_CUDA(cudaFree(d_diag));
     CHECK_CUDA(cudaFree(d_V_R));
     CHECK_CUDA(cudaFree(d_Gram_dp));
     if (d_W_dp) CHECK_CUDA(cudaFree(d_W_dp));
     if (d_work_dp) CHECK_CUDA(cudaFree(d_work_dp));
     if (d_info) CHECK_CUDA(cudaFree(d_info));
   } else {
    const double eps_mach_dp = 1.1102230246251565e-16;
    double alpha_dp = 1.0, beta_zero_dp = 0.0;

    double* d_A_dp;
    CHECK_CUDA(cudaMalloc(&d_A_dp, sizeof(double) * (long long)M * N));
    convert_scalar_to_dp<<<((long long)M*N + 255)/256, 256>>>(d_A, d_A_dp, (long long)M * N);

    double* d_Gram_dp;
    cudaEventRecord(ev_start);
    CHECK_CUDA(cudaMalloc(&d_Gram_dp, sizeof(double) * M * M));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  AA^T alloc (dp): " << ev_ms << " ms\n";

    cudaEventRecord(ev_start);
    CHECK_CUBLAS(cublasDsyrk(cublasH, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
        M, N, &alpha_dp, d_A_dp, N, &beta_zero_dp, d_Gram_dp, M));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  AA^T syrk (dp): " << ev_ms << " ms\n";
    CHECK_CUDA(cudaFree(d_A_dp));

    // trace(AA^T) = ||A||_F^2  — dasum on diagonal (all entries >= 0 for Gram)
    double traceAAT;
    CHECK_CUBLAS(cublasDasum(cublasH, M, d_Gram_dp, M + 1, &traceAAT));
    if (verbose) std::cout << "  trace(AAT): " << traceAAT << "\n";

    double *d_W_dp;
    CHECK_CUDA(cudaMalloc(&d_W_dp, sizeof(double) * M));
    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER, M, d_Gram_dp, M, d_W_dp, &lwork));

    double *d_work_dp; int *d_info;
    cudaEventRecord(ev_start);
    CHECK_CUDA(cudaMalloc(&d_work_dp, sizeof(double) * lwork));
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  eig buf alloc: " << ev_ms << " ms\n";

    cudaEventRecord(ev_start);
    CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER, M, d_Gram_dp, M, d_W_dp, d_work_dp, lwork, d_info));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  eig decomp (dp): " << ev_ms << " ms\n";

    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::cerr << "  syevd failed: devInfo = " << h_info << "\n";
    }

    CHECK_CUDA(cudaMemset(d_info, 0, sizeof(int)));
    if (R > 1)
    check_resolution_dp<<<1, R - 1>>>(d_W_dp, N + M, M, R, eps_mach_dp, d_info, traceAAT);
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
      if (verbose)
        std::cout << "  resolution condition failed among top-" << R
                  << " eigenvalues, falling back to full SVD\n";
      CHECK_CUDA(cudaFree(d_Gram_dp)); CHECK_CUDA(cudaFree(d_W_dp));
      CHECK_CUDA(cudaFree(d_work_dp)); CHECK_CUDA(cudaFree(d_info));
      gpu_full_svd_update_factor(cusolverH, cublasH, d_A, M, N, R, d_factor, verbose);
      return;
    }

    // Convert top-R eigenvectors double→float, then transpose
    scalar_t* d_eigvecs;
    CHECK_CUDA(cudaMalloc(&d_eigvecs, sizeof(scalar_t) * M * R));
    convert_dp_to_scalar<<<((long long)M*R + 255)/256, 256>>>(
      d_Gram_dp + (long long)(M - R) * M, d_eigvecs, (long long)M * R);

    cudaEventRecord(ev_start);
    { scalar_t one = (scalar_t)1, zero = (scalar_t)0;
      CHECK_CUBLAS(cublasGeamT(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
          R, M,
          &one,  d_eigvecs, M,
          &zero, d_factor, R,
          d_factor, R));
    }
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  transpose eigvecs: " << ev_ms << " ms\n";

    CHECK_CUDA(cudaFree(d_eigvecs));
    CHECK_CUDA(cudaFree(d_Gram_dp)); CHECK_CUDA(cudaFree(d_W_dp));
    CHECK_CUDA(cudaFree(d_work_dp)); CHECK_CUDA(cudaFree(d_info));
}
#endif
   CHECK_CUDA(cudaDeviceSynchronize());
   cudaEventDestroy(ev_start);
   cudaEventDestroy(ev_stop);
}

#if !SCALAR_DOUBLE
static void gpu_full_svd_update_factor_pinned_workspace(
  cusolverDnHandle_t cusolverH,
  cublasHandle_t cublasH,
  scalar_t* d_A,
  int M,
  int N,
  int R,
  scalar_t* d_factor,
  const PinnedBlockedFullSVDPlan& plan,
  PinnedBlockedFullSVDWorkspace& ws,
  bool verbose)
{
  int min_mn = std::min(M, N);
  R = std::min(R, min_mn);
  cudaEvent_t ev0, ev1;
  float ev_ms = 0.f;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);

  if (!(M > N) || !plan.active) {
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    throw std::runtime_error("Pinned full SVD fallback requires an active M>N plan.");
  }
  if (!ws.d_full_s || !ws.d_full_u || !ws.d_full_colmaj_A || !ws.d_full_vt_dummy || !ws.d_info) {
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    throw std::runtime_error("Pinned full SVD fallback workspace is not initialized.");
  }

  scalar_t* d_S = ws.d_full_s;
  scalar_t* d_U = ws.d_full_u;
  scalar_t* d_VT_dummy = ws.d_full_vt_dummy;
  scalar_t* d_colmaj_A = ws.d_full_colmaj_A;
  int* d_info = ws.d_info;
  int lwork = plan.full_svd_lwork;
  scalar_t* d_work = ws.d_full_work;
  bool free_local_work = false;
  if (!d_work || lwork <= 0) {
    lwork = 0;
    CHECK_CUSOLVER(cusolverGesvdBufSizeT(cusolverH, M, N, &lwork));
    CHECK_CUDA(cudaMalloc(&d_work, sizeof(scalar_t) * std::max(lwork, 1)));
    free_local_work = true;
  }

  scalar_t one = (scalar_t)1;
  scalar_t zero = (scalar_t)0;
  CHECK_CUBLAS(cublasGeamT(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, M, N,
    &one, d_A, N, &zero, d_colmaj_A, M, d_colmaj_A, M));

  CHECK_CUDA(cudaMemset(d_info, 0, sizeof(int)));
  cudaEventRecord(ev0);
  CHECK_CUSOLVER(cusolverGesvdT(cusolverH, 'S', 'N', M, N, d_colmaj_A, M,
    d_S, d_U, M, d_VT_dummy, 1, d_work, lwork, nullptr, d_info));
  cudaEventRecord(ev1);
  cudaEventSynchronize(ev1);
  cudaEventElapsedTime(&ev_ms, ev0, ev1);
  if (verbose) std::cout << "  cusolverDnGesvd: " << ev_ms << " ms\n";

  int h_info = 0;
  CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info != 0)
    std::cerr << "  WARNING: cusolverDnGesvd info=" << h_info << "\n";

  CHECK_CUBLAS(cublasGeamT(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R, M,
    &one, d_U, M, &zero, d_factor, R, d_factor, R));

  if (free_local_work) CHECK_CUDA(cudaFree(d_work));
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
}

static void gpu_truncated_svd_update_factor_pinned_blocked(
  cusolverDnHandle_t cusolverH,
  cublasHandle_t cublasH,
  scalar_t* d_A,
  int M,
  int N,
  int R,
  scalar_t* d_factor,
  const PinnedBlockedFullSVDPlan& plan,
  PinnedBlockedFullSVDWorkspace& ws,
  bool verbose)
{
  scalar_t alpha = (scalar_t)1, beta = (scalar_t)0;
  const double eps_mach_dp = 1.1102230246251565e-16;
  const double alpha_dp = 1.0;
  const double beta_zero = 0.0;
  const double beta_one = 1.0;
  cudaEvent_t ev_start, ev_stop;
  float ev_ms = 0.f;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  int K = std::min(M, N);
  R = std::min(R, K);
  if (!(M > N) || !plan.active) {
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    throw std::runtime_error("Pinned blocked full SVD path requires an active M>N plan.");
  }
  if (!ws.d_gram_dp || !ws.d_blk_dp || !ws.d_W_dp || !ws.d_work_dp || !ws.d_info) {
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    throw std::runtime_error("Pinned blocked full SVD workspace is not initialized.");
  }

  double* d_Gram_dp = ws.d_gram_dp;
  double* d_blk_dp = ws.d_blk_dp;
  double* d_W_dp = ws.d_W_dp;
  double* d_work_dp = ws.d_work_dp;
  int* d_info = ws.d_info;

  cudaEventRecord(ev_start);
  for (int s = 0; s < M; s += plan.block_rows) {
    int b = std::min(plan.block_rows, M - s);
    long long chunk_elems = (long long)b * N;
    convert_scalar_to_dp<<<(chunk_elems + 255) / 256, 256>>>(
      d_A + (long long)s * N, d_blk_dp, chunk_elems);
    CHECK_CUDA(cudaGetLastError());
    if (plan.use_syrk) {
      CHECK_CUBLAS(cublasDsyrk(cublasH, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        N, b, &alpha_dp, d_blk_dp, N,
        (s == 0) ? &beta_zero : &beta_one, d_Gram_dp, N));
    } else {
      CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
        N, N, b, &alpha_dp, d_blk_dp, N, d_blk_dp, N,
        (s == 0) ? &beta_zero : &beta_one, d_Gram_dp, N));
    }
  }
  cudaEventRecord(ev_stop);
  cudaEventSynchronize(ev_stop);
  cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
  if (verbose) std::cout << "  ATA " << (plan.use_syrk ? "syrk" : "gemm") << " (dp): " << ev_ms << " ms\n";

  double traceATA = 0.0;
  CHECK_CUBLAS(cublasDasum(cublasH, N, d_Gram_dp, N + 1, &traceATA));
  if (verbose) std::cout << "  trace(ATA): " << traceATA << "\n";

  CHECK_CUDA(cudaMemset(d_info, 0, sizeof(int)));
  cudaEventRecord(ev_start);
  cusolverStatus_t st = cusolverDnDsyevd(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER, N, d_Gram_dp, N, d_W_dp, d_work_dp, plan.lwork, d_info);
  cudaEventRecord(ev_stop);
  cudaEventSynchronize(ev_stop);
  cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
  if (verbose) std::cout << "  eig decomp (dp): " << ev_ms << " ms\n";
  if (st != CUSOLVER_STATUS_SUCCESS) {
    if (verbose) {
      std::cout << "  pinned blocked eigensolver failed with status="
                << cusolverStatusString(st)
                << ", falling back to pinned full SVD\n";
    }
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    gpu_full_svd_update_factor_pinned_workspace(cusolverH, cublasH, d_A, M, N, R, d_factor, plan, ws, verbose);
    return;
  }

  int h_info = 0;
  CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info != 0) {
    if (verbose) {
      std::cout << "  pinned blocked eigensolver returned info=" << h_info
                << ", falling back to pinned full SVD\n";
    }
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    gpu_full_svd_update_factor_pinned_workspace(cusolverH, cublasH, d_A, M, N, R, d_factor, plan, ws, verbose);
    return;
  }

  CHECK_CUDA(cudaMemset(d_info, 0, sizeof(int)));
  if (R > 1)
    check_resolution_dp<<<1, R - 1>>>(d_W_dp, M + N, N, R, eps_mach_dp, d_info, traceATA);
  CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info != 0) {
    if (verbose)
      std::cout << "  resolution condition failed among top-" << R
                << " eigenvalues, falling back to full SVD\n";
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    gpu_full_svd_update_factor_pinned_workspace(cusolverH, cublasH, d_A, M, N, R, d_factor, plan, ws, verbose);
    return;
  }

  scalar_t* d_V_R = nullptr;
  scalar_t* d_diag = nullptr;
  double* d_V_R_dp = d_Gram_dp + (long long)(N - R) * N;
  CHECK_CUDA(cudaMalloc(&d_V_R, sizeof(scalar_t) * (long long)N * R));
  convert_dp_to_scalar<<<((long long)N * R + 255) / 256, 256>>>(d_V_R_dp, d_V_R, (long long)N * R);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMalloc(&d_diag, sizeof(scalar_t) * R));
  compute_inv_sigma_dp<<<(R + 255) / 256, 256>>>(d_W_dp, d_diag, N, R, 1e-12);
  CHECK_CUDA(cudaGetLastError());

  cudaEventRecord(ev_start);
  CHECK_CUBLAS(cublasGemmT(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
    R, M, N, &alpha, d_V_R, N, d_A, N, &beta, d_factor, R));
  cudaEventRecord(ev_stop);
  cudaEventSynchronize(ev_stop);
  cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
  if (verbose) std::cout << "  AV gemm: " << ev_ms << " ms\n";

  cudaEventRecord(ev_start);
  CHECK_CUBLAS(cublasDgmmT(cublasH, CUBLAS_SIDE_LEFT, R, M,
    d_factor, R, d_diag, 1, d_factor, R));
  cudaEventRecord(ev_stop);
  cudaEventSynchronize(ev_stop);
  cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
  if (verbose) std::cout << "  normalize (1/sigma): " << ev_ms << " ms\n";

  CHECK_CUDA(cudaFree(d_diag));
  CHECK_CUDA(cudaFree(d_V_R));
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);
}
#endif

 
 
 
 // ===================================================================
 // COO tensor reader (from .tns file)
 // Format: line 1 = order, line 2 = space-separated dims,
 //         rest = 1-based indices + float value per non-zero
 // ===================================================================
 struct COOTensor {
   int order = 0;
   std::vector<uint64_t> dims;
   std::vector<std::vector<uint64_t>> indices;  // [nnz][mode], 0-based
   std::vector<scalar_t> values;
 };
 
 COOTensor readCOOTensor(const std::string& filename) {
   std::ifstream inFile(filename);
   if (!inFile) throw std::runtime_error("Cannot open COO file: " + filename);
   COOTensor coo;
   std::string line;
   bool orderRead = false, dimsRead = false;
   while (std::getline(inFile, line)) {
     if (line.empty() || line[0] == '#') continue;
     if (!orderRead) {
       std::istringstream iss(line); iss >> coo.order;
       coo.dims.resize(coo.order); orderRead = true; continue;
     }
     if (!dimsRead) {
       std::istringstream iss(line);
       for (int i = 0; i < coo.order; i++) iss >> coo.dims[i];
       dimsRead = true; continue;
     }
     std::istringstream iss(line);
     std::vector<uint64_t> idx(coo.order);
     scalar_t val; bool ok = true;
     for (int i = 0; i < coo.order; i++) {
       if (!(iss >> idx[i])) { ok = false; break; }
       idx[i]--;  // 1-based → 0-based
     }
     if (!ok || !(iss >> val)) continue;
     coo.indices.push_back(idx);
     coo.values.push_back(val);
   }
   std::cout << "Read COO tensor: order=" << coo.order << ", nnz=" << coo.values.size() << "\n";
   return coo;
 }
 
 
 // ===================================================================
 // CSF copy: modeOrder[0] = root (= non-contracting mode for TTMc-n),
 // modeOrder[1..] = remaining modes sorted ascending by unique-index count.
 // GPU pointers are null until uploadCSFToGPU() is called.
 // ===================================================================
 struct CSFCopy {
   int order = 3;
   std::vector<int>      modeOrder;
   std::vector<uint64_t> dims;
   std::vector<std::vector<uint64_t>> ptrs;
   std::vector<std::vector<uint64_t>> idxs;
   std::vector<scalar_t> values;
 
   std::vector<uint64_t*> d_ptrs;   // d_ptrs[l] = GPU ptr array for level l
   std::vector<uint64_t*> d_idxs;   // d_idxs[l] = GPU idx array for level l
   scalar_t  *d_values = nullptr;
 };
 
 CSFCopy buildCSFCopy(const COOTensor& coo, int rootMode,
                      const std::vector<size_t>& uniqueCounts) {
   const int order = coo.order;
   size_t nnz = coo.values.size();
 
   std::vector<int> rest;
   for (int m = 0; m < order; m++) if (m != rootMode) rest.push_back(m);
   std::sort(rest.begin(), rest.end(),
             [&](int a, int b){ return uniqueCounts[a] < uniqueCounts[b]; });
 
   CSFCopy csf;
   csf.order = order;
   csf.modeOrder.clear();
   csf.modeOrder.push_back(rootMode);
   for (int r : rest) csf.modeOrder.push_back(r);
   csf.dims.resize(order);
   for (int l = 0; l < order; l++) csf.dims[l] = coo.dims[csf.modeOrder[l]];
 
   std::vector<size_t> perm(nnz);
   std::iota(perm.begin(), perm.end(), 0);
   std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
     for (int l = 0; l < order; l++) {
       int m = csf.modeOrder[l];
       if (coo.indices[a][m] != coo.indices[b][m])
         return coo.indices[a][m] < coo.indices[b][m];
     }
     return false;
   });
 
   csf.ptrs.resize(order);
   csf.idxs.resize(order);
   csf.ptrs[0].push_back(0);
   std::vector<uint64_t> prev(order, UINT64_MAX);
 
   for (size_t pi = 0; pi < nnz; pi++) {
     size_t ei = perm[pi];
     std::vector<uint64_t> cur(order);
     for (int l = 0; l < order; l++)
       cur[l] = coo.indices[ei][csf.modeOrder[l]];
     bool changed = false;
     for (int l = 0; l < order; l++) {
       if (cur[l] != prev[l] || changed) {
         changed = true;
         csf.idxs[l].push_back(cur[l]);
         if (l + 1 < order) csf.ptrs[l + 1].push_back(csf.idxs[l + 1].size());
         prev[l] = cur[l];
       }
     }
     csf.values.push_back(coo.values[ei]);
   }
   for (int l = 0; l < order; l++) csf.ptrs[l].push_back(csf.idxs[l].size());
   return csf;
 }
 
 void uploadCSFToGPU(CSFCopy& csf) {
   int order = csf.order;
   csf.d_ptrs.assign(order, nullptr);
   csf.d_idxs.assign(order, nullptr);
   // Level 0: idx only (root has no meaningful ptr)
   CHECK_CUDA(cudaMalloc(&csf.d_idxs[0], sizeof(uint64_t) * csf.idxs[0].size()));
   CHECK_CUDA(cudaMemcpy(csf.d_idxs[0], csf.idxs[0].data(),
     sizeof(uint64_t) * csf.idxs[0].size(), cudaMemcpyHostToDevice));
   // Levels 1..order-1: ptr and idx
   for (int l = 1; l < order; l++) {
     CHECK_CUDA(cudaMalloc(&csf.d_ptrs[l], sizeof(uint64_t) * csf.ptrs[l].size()));
     CHECK_CUDA(cudaMemcpy(csf.d_ptrs[l], csf.ptrs[l].data(),
       sizeof(uint64_t) * csf.ptrs[l].size(), cudaMemcpyHostToDevice));
     CHECK_CUDA(cudaMalloc(&csf.d_idxs[l], sizeof(uint64_t) * csf.idxs[l].size()));
     CHECK_CUDA(cudaMemcpy(csf.d_idxs[l], csf.idxs[l].data(),
       sizeof(uint64_t) * csf.idxs[l].size(), cudaMemcpyHostToDevice));
   }
   CHECK_CUDA(cudaMalloc(&csf.d_values, sizeof(scalar_t) * csf.values.size()));
   CHECK_CUDA(cudaMemcpy(csf.d_values, csf.values.data(),
     sizeof(scalar_t) * csf.values.size(), cudaMemcpyHostToDevice));
 }
 
void freeCSFGPU(CSFCopy& csf) {
  for (auto& p : csf.d_ptrs) { if (p) { CHECK_CUDA(cudaFree(p)); p = nullptr; } }
  for (auto& p : csf.d_idxs) { if (p) { CHECK_CUDA(cudaFree(p)); p = nullptr; } }
  csf.d_ptrs.clear();
  csf.d_idxs.clear();
  if (csf.d_values) { CHECK_CUDA(cudaFree(csf.d_values)); csf.d_values = nullptr; }
}

enum class TTMcStorageMode {
  Auto,
  Full,
  Tiled
};

static const char* ttmcStorageModeName(TTMcStorageMode mode) {
  switch (mode) {
    case TTMcStorageMode::Auto:  return "auto";
    case TTMcStorageMode::Full:  return "full";
    case TTMcStorageMode::Tiled: return "tiled";
  }
  return "unknown";
}

static TTMcStorageMode parseTTMcStorageMode(const std::string& value) {
  if (value == "auto")  return TTMcStorageMode::Auto;
  if (value == "full")  return TTMcStorageMode::Full;
  if (value == "tiled") return TTMcStorageMode::Tiled;
  throw std::runtime_error("Invalid --ttmc-path value. Expected auto|full|tiled.");
}

static std::string formatBytes(double bytes) {
  static const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  int unit = 0;
  while (bytes >= 1024.0 && unit < 4) {
    bytes /= 1024.0;
    ++unit;
  }
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision((unit == 0) ? 0 : 2);
  oss << bytes << " " << units[unit];
  return oss.str();
}

static uint64_t chooseTileRootCount(
  uint64_t num_roots,
  uint64_t rank_product,
  uint64_t rank,
  uint64_t override_roots,
  size_t target_bytes)
{
  if (num_roots == 0) return 0;
  if (override_roots > 0) return std::min(override_roots, num_roots);
  uint64_t bytes_per_row = std::max<uint64_t>(1, rank_product + rank) * sizeof(scalar_t);
  uint64_t rows = std::max<uint64_t>(1, (uint64_t)target_bytes / std::max<uint64_t>(1, bytes_per_row));
  return std::min(rows, num_roots);
}

static size_t tiledModeWorkspaceBytes(
  uint64_t tile_rows,
  uint64_t rank_product,
  uint64_t rank)
{
  size_t bytes = sizeof(scalar_t) * tile_rows * (size_t)rank_product;
  bytes += sizeof(scalar_t) * tile_rows * (size_t)rank;
#if !SCALAR_DOUBLE
  bytes += sizeof(double) * std::min<uint64_t>(tile_rows, 8192ULL) * (size_t)rank_product;
#endif
  return bytes;
}


struct TiledGramSVDResult {
  scalar_t* d_VR = nullptr;        // col-major (P x R)
  scalar_t* d_inv_sigma = nullptr; // length R
  double core_norm_sq = 0.0;
  int cols = 0;
  int rank = 0;
};

static void freeTiledGramSVDResult(TiledGramSVDResult& result) {
  if (result.d_VR) {
    CHECK_CUDA(cudaFree(result.d_VR));
    result.d_VR = nullptr;
  }
  if (result.d_inv_sigma) {
    CHECK_CUDA(cudaFree(result.d_inv_sigma));
    result.d_inv_sigma = nullptr;
  }
  result.core_norm_sq = 0.0;
  result.cols = 0;
  result.rank = 0;
}

struct TTMCTile3D {
  uint64_t i_ptr_begin = 0;
  uint64_t num_rows = 0;
  uint64_t num_j = 0;
  uint64_t num_k = 0;
  uint64_t num_l = 0;
  uint64_t output_elems = 0;

  std::vector<uint64_t> h_local_rows;
  std::vector<uint64_t> h_mode1_ptr;
  std::vector<uint64_t> h_mode2_ptr;
  std::vector<uint64_t> h_mode3_ptr;

  uint64_t* d_local_rows = nullptr;
  uint64_t* d_mode1_ptr = nullptr;
  uint64_t* d_mode2_ptr = nullptr;
  uint64_t* d_mode3_ptr = nullptr;

  uint64_t* d_global_rows = nullptr; // borrowed from csf.d_idxs[0]
  scalar_t* d_values = nullptr;      // borrowed from csf.d_values

  std::vector<uint64_t*> d_ptrs;
  std::vector<uint64_t*> d_idxs;
  TTMcCache cache;
};

struct TiledModeWorkspace {
  scalar_t* d_y_tile = nullptr;
  scalar_t* d_factor_tile = nullptr;
#if !SCALAR_DOUBLE
  double* d_tile_block_dp = nullptr;
  int dp_block_rows = 0;
#endif
  uint64_t y_capacity_elems = 0;
  uint64_t factor_capacity_elems = 0;
#if !SCALAR_DOUBLE
  uint64_t dp_capacity_elems = 0;
#endif
};

struct TiledModeUpdateStats {
  double total_us = 0.0;
  double ttmc_us = 0.0;
  double core_norm_sq = 0.0;
};

__global__ void scatter_factor_rows_kernel(
  const uint64_t* __restrict__ global_rows,
  const scalar_t* __restrict__ tile_rows,
  scalar_t* __restrict__ factor,
  uint64_t num_rows,
  int R)
{
  uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t total = num_rows * (uint64_t)R;
  if (idx >= total) return;
  uint64_t row = idx / (uint64_t)R;
  int col = (int)(idx % (uint64_t)R);
  uint64_t global_row = global_rows[row];
  factor[global_row * (uint64_t)R + col] = tile_rows[row * (uint64_t)R + col];
}

static void freeTiledModeWorkspace(TiledModeWorkspace& ws) {
  if (ws.d_y_tile) {
    CHECK_CUDA(cudaFree(ws.d_y_tile));
    ws.d_y_tile = nullptr;
  }
  if (ws.d_factor_tile) {
    CHECK_CUDA(cudaFree(ws.d_factor_tile));
    ws.d_factor_tile = nullptr;
  }
#if !SCALAR_DOUBLE
  if (ws.d_tile_block_dp) {
    CHECK_CUDA(cudaFree(ws.d_tile_block_dp));
    ws.d_tile_block_dp = nullptr;
  }
  ws.dp_block_rows = 0;
  ws.dp_capacity_elems = 0;
#endif
  ws.y_capacity_elems = 0;
  ws.factor_capacity_elems = 0;
}

static uint64_t tiledWorkspaceYElems(uint64_t max_rows, int cols)
{
  return max_rows * (uint64_t)std::max(cols, 0);
}

static uint64_t tiledWorkspaceFactorElems(uint64_t max_rows, int rank)
{
  return max_rows * (uint64_t)std::max(rank, 0);
}

#if !SCALAR_DOUBLE
static uint64_t tiledWorkspaceDPElems(uint64_t max_rows, int cols)
{
  return std::min<uint64_t>(max_rows, 8192ULL) * (uint64_t)std::max(cols, 0);
}
#endif

static bool tryPrepareTiledModeWorkspace(
  TiledModeWorkspace& ws,
  uint64_t y_capacity_elems,
  uint64_t factor_capacity_elems,
#if !SCALAR_DOUBLE
  uint64_t dp_capacity_elems
#endif
)
{
  freeTiledModeWorkspace(ws);
  ws.y_capacity_elems = y_capacity_elems;
  ws.factor_capacity_elems = factor_capacity_elems;
  if (y_capacity_elems == 0 && factor_capacity_elems == 0
#if !SCALAR_DOUBLE
      && dp_capacity_elems == 0
#endif
  ) {
    return true;
  }

  cudaError_t err = cudaSuccess;
  if (y_capacity_elems > 0) {
    err = cudaMalloc(&ws.d_y_tile, sizeof(scalar_t) * y_capacity_elems);
    if (err != cudaSuccess) {
      cudaGetLastError();
      freeTiledModeWorkspace(ws);
      return false;
    }
  }
  if (factor_capacity_elems > 0) {
    err = cudaMalloc(&ws.d_factor_tile, sizeof(scalar_t) * factor_capacity_elems);
    if (err != cudaSuccess) {
      cudaGetLastError();
      freeTiledModeWorkspace(ws);
      return false;
    }
  }
#if !SCALAR_DOUBLE
  ws.dp_capacity_elems = dp_capacity_elems;
  ws.dp_block_rows = (dp_capacity_elems > 0) ? 8192 : 0;
  if (dp_capacity_elems > 0) {
    err = cudaMalloc(&ws.d_tile_block_dp, sizeof(double) * dp_capacity_elems);
    if (err != cudaSuccess) {
      cudaGetLastError();
      freeTiledModeWorkspace(ws);
      return false;
    }
  }
#endif
  return true;
}

static bool tryAllocateTiledPass1AndGram(
  uint64_t tile_rows,
  uint64_t rank_product)
{
  TiledModeWorkspace probe_ws;
  bool ws_ok = tryPrepareTiledModeWorkspace(
    probe_ws,
    tiledWorkspaceYElems(tile_rows, (int)rank_product),
    /*factor_capacity_elems=*/0,
#if !SCALAR_DOUBLE
    tiledWorkspaceDPElems(tile_rows, (int)rank_product)
#endif
  );
  if (!ws_ok) {
    freeTiledModeWorkspace(probe_ws);
    return false;
  }

  void* d_gram_probe = nullptr;
#if SCALAR_DOUBLE
  cudaError_t gram_err = cudaMalloc(&d_gram_probe, sizeof(scalar_t) * rank_product * rank_product);
#else
  cudaError_t gram_err = cudaMalloc(&d_gram_probe, sizeof(double) * rank_product * rank_product);
#endif
  bool ok = (gram_err == cudaSuccess);
  if (!ok) cudaGetLastError();
  if (d_gram_probe) CHECK_CUDA(cudaFree(d_gram_probe));
  freeTiledModeWorkspace(probe_ws);
  return ok;
}

static void freeTTMCTile3D(TTMCTile3D& tile) {
  free_ttmc_cache(tile.cache);
  if (tile.d_local_rows) {
    CHECK_CUDA(cudaFree(tile.d_local_rows));
    tile.d_local_rows = nullptr;
  }
  if (tile.d_mode1_ptr) {
    CHECK_CUDA(cudaFree(tile.d_mode1_ptr));
    tile.d_mode1_ptr = nullptr;
  }
  if (tile.d_mode2_ptr) {
    CHECK_CUDA(cudaFree(tile.d_mode2_ptr));
    tile.d_mode2_ptr = nullptr;
  }
  if (tile.d_mode3_ptr) {
    CHECK_CUDA(cudaFree(tile.d_mode3_ptr));
    tile.d_mode3_ptr = nullptr;
  }
  tile.d_global_rows = nullptr;
  tile.d_values = nullptr;
  tile.d_ptrs.clear();
  tile.d_idxs.clear();
  tile.h_local_rows.clear();
  tile.h_mode1_ptr.clear();
  tile.h_mode2_ptr.clear();
  tile.h_mode3_ptr.clear();
  tile.num_l = 0;
}

static bool buildTTMCTile3D(
  const CSFCopy& csf,
  uint64_t i_ptr_begin,
  uint64_t tile_rows,
  uint64_t* ranks_v,
  uint64_t* factor_sizes_v,
  uint64_t* dims_v,
  TTMCTile3D& tile)
{
  if (csf.order != 3) return false;
  uint64_t num_roots = (uint64_t)csf.idxs[0].size();
  if (i_ptr_begin >= num_roots) return false;

  tile.i_ptr_begin = i_ptr_begin;
  tile.num_rows = std::min(tile_rows, num_roots - i_ptr_begin);
  if (tile.num_rows == 0) return false;

  uint64_t j_begin = csf.ptrs[1][i_ptr_begin];
  uint64_t j_end = csf.ptrs[1][i_ptr_begin + tile.num_rows];
  uint64_t k_begin = csf.ptrs[2][j_begin];
  uint64_t k_end = csf.ptrs[2][j_end];

  tile.num_j = j_end - j_begin;
  tile.num_k = k_end - k_begin;
  tile.num_l = tile.num_k;
  tile.output_elems = tile.num_rows * ranks_v[1] * ranks_v[2];

  tile.h_local_rows.resize(tile.num_rows);
  for (uint64_t i = 0; i < tile.num_rows; ++i) tile.h_local_rows[i] = i;

  tile.h_mode1_ptr.resize(tile.num_rows + 1);
  for (uint64_t i = 0; i <= tile.num_rows; ++i)
    tile.h_mode1_ptr[i] = csf.ptrs[1][i_ptr_begin + i] - j_begin;

  tile.h_mode2_ptr.resize(tile.num_j + 1);
  for (uint64_t j = 0; j <= tile.num_j; ++j)
    tile.h_mode2_ptr[j] = csf.ptrs[2][j_begin + j] - k_begin;

  CHECK_CUDA(cudaMalloc(&tile.d_local_rows, sizeof(uint64_t) * tile.num_rows));
  CHECK_CUDA(cudaMemcpy(
    tile.d_local_rows,
    tile.h_local_rows.data(),
    sizeof(uint64_t) * tile.num_rows,
    cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&tile.d_mode1_ptr, sizeof(uint64_t) * tile.h_mode1_ptr.size()));
  CHECK_CUDA(cudaMemcpy(
    tile.d_mode1_ptr,
    tile.h_mode1_ptr.data(),
    sizeof(uint64_t) * tile.h_mode1_ptr.size(),
    cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&tile.d_mode2_ptr, sizeof(uint64_t) * tile.h_mode2_ptr.size()));
  CHECK_CUDA(cudaMemcpy(
    tile.d_mode2_ptr,
    tile.h_mode2_ptr.data(),
    sizeof(uint64_t) * tile.h_mode2_ptr.size(),
    cudaMemcpyHostToDevice));

  tile.d_ptrs.assign(3, nullptr);
  tile.d_idxs.assign(3, nullptr);
  tile.d_ptrs[1] = tile.d_mode1_ptr;
  tile.d_ptrs[2] = tile.d_mode2_ptr;
  tile.d_idxs[0] = tile.d_local_rows;
  tile.d_idxs[1] = csf.d_idxs[1] + j_begin;
  tile.d_idxs[2] = csf.d_idxs[2] + k_begin;
  tile.d_global_rows = csf.d_idxs[0] + i_ptr_begin;
  tile.d_values = csf.d_values + k_begin;

  uint64_t* mode_ptrs[3] = {nullptr, tile.h_mode1_ptr.data(), tile.h_mode2_ptr.data()};
  uint64_t* mode_idxs[3] = {
    tile.h_local_rows.data(),
    const_cast<uint64_t*>(csf.idxs[1].data() + j_begin),
    const_cast<uint64_t*>(csf.idxs[2].data() + k_begin)
  };
  uint64_t size_mode_ptr[3] = {0, (uint64_t)tile.h_mode1_ptr.size(), (uint64_t)tile.h_mode2_ptr.size()};
  uint64_t size_mode_idx[3] = {tile.num_rows, tile.num_j, tile.num_k};

  prepare_ttmc_cuda(
    mode_ptrs, mode_idxs,
    size_mode_ptr, size_mode_idx,
    ranks_v, factor_sizes_v, dims_v,
    /*ncm=*/0, /*order=*/3, tile.cache);
  return true;
}

static bool buildTTMCTile4D(
  const CSFCopy& csf,
  uint64_t i_ptr_begin,
  uint64_t tile_rows,
  uint64_t* ranks_v,
  uint64_t* factor_sizes_v,
  uint64_t* dims_v,
  TTMCTile3D& tile)
{
  if (csf.order != 4) return false;
  uint64_t num_roots = (uint64_t)csf.idxs[0].size();
  if (i_ptr_begin >= num_roots) return false;

  tile.i_ptr_begin = i_ptr_begin;
  tile.num_rows = std::min(tile_rows, num_roots - i_ptr_begin);
  if (tile.num_rows == 0) return false;

  uint64_t j_begin = csf.ptrs[1][i_ptr_begin];
  uint64_t j_end = csf.ptrs[1][i_ptr_begin + tile.num_rows];
  uint64_t k_begin = csf.ptrs[2][j_begin];
  uint64_t k_end = csf.ptrs[2][j_end];
  uint64_t l_begin = csf.ptrs[3][k_begin];
  uint64_t l_end = csf.ptrs[3][k_end];

  tile.num_j = j_end - j_begin;
  tile.num_k = k_end - k_begin;
  tile.num_l = l_end - l_begin;
  tile.output_elems = tile.num_rows * ranks_v[1] * ranks_v[2] * ranks_v[3];

  tile.h_local_rows.resize(tile.num_rows);
  for (uint64_t i = 0; i < tile.num_rows; ++i) tile.h_local_rows[i] = i;

  tile.h_mode1_ptr.resize(tile.num_rows + 1);
  for (uint64_t i = 0; i <= tile.num_rows; ++i)
    tile.h_mode1_ptr[i] = csf.ptrs[1][i_ptr_begin + i] - j_begin;

  tile.h_mode2_ptr.resize(tile.num_j + 1);
  for (uint64_t j = 0; j <= tile.num_j; ++j)
    tile.h_mode2_ptr[j] = csf.ptrs[2][j_begin + j] - k_begin;

  tile.h_mode3_ptr.resize(tile.num_k + 1);
  for (uint64_t k = 0; k <= tile.num_k; ++k)
    tile.h_mode3_ptr[k] = csf.ptrs[3][k_begin + k] - l_begin;

  CHECK_CUDA(cudaMalloc(&tile.d_local_rows, sizeof(uint64_t) * tile.num_rows));
  CHECK_CUDA(cudaMemcpy(
    tile.d_local_rows,
    tile.h_local_rows.data(),
    sizeof(uint64_t) * tile.num_rows,
    cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&tile.d_mode1_ptr, sizeof(uint64_t) * tile.h_mode1_ptr.size()));
  CHECK_CUDA(cudaMemcpy(
    tile.d_mode1_ptr,
    tile.h_mode1_ptr.data(),
    sizeof(uint64_t) * tile.h_mode1_ptr.size(),
    cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&tile.d_mode2_ptr, sizeof(uint64_t) * tile.h_mode2_ptr.size()));
  CHECK_CUDA(cudaMemcpy(
    tile.d_mode2_ptr,
    tile.h_mode2_ptr.data(),
    sizeof(uint64_t) * tile.h_mode2_ptr.size(),
    cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&tile.d_mode3_ptr, sizeof(uint64_t) * tile.h_mode3_ptr.size()));
  CHECK_CUDA(cudaMemcpy(
    tile.d_mode3_ptr,
    tile.h_mode3_ptr.data(),
    sizeof(uint64_t) * tile.h_mode3_ptr.size(),
    cudaMemcpyHostToDevice));

  tile.d_ptrs.assign(4, nullptr);
  tile.d_idxs.assign(4, nullptr);
  tile.d_ptrs[1] = tile.d_mode1_ptr;
  tile.d_ptrs[2] = tile.d_mode2_ptr;
  tile.d_ptrs[3] = tile.d_mode3_ptr;
  tile.d_idxs[0] = tile.d_local_rows;
  tile.d_idxs[1] = csf.d_idxs[1] + j_begin;
  tile.d_idxs[2] = csf.d_idxs[2] + k_begin;
  tile.d_idxs[3] = csf.d_idxs[3] + l_begin;
  tile.d_global_rows = csf.d_idxs[0] + i_ptr_begin;
  tile.d_values = csf.d_values + l_begin;

  uint64_t* mode_ptrs[4] = {
    nullptr,
    tile.h_mode1_ptr.data(),
    tile.h_mode2_ptr.data(),
    tile.h_mode3_ptr.data()
  };
  uint64_t* mode_idxs[4] = {
    tile.h_local_rows.data(),
    const_cast<uint64_t*>(csf.idxs[1].data() + j_begin),
    const_cast<uint64_t*>(csf.idxs[2].data() + k_begin),
    const_cast<uint64_t*>(csf.idxs[3].data() + l_begin)
  };
  uint64_t size_mode_ptr[4] = {
    0,
    (uint64_t)tile.h_mode1_ptr.size(),
    (uint64_t)tile.h_mode2_ptr.size(),
    (uint64_t)tile.h_mode3_ptr.size()
  };
  uint64_t size_mode_idx[4] = {
    tile.num_rows,
    tile.num_j,
    tile.num_k,
    tile.num_l
  };

  prepare_ttmc_cuda(
    mode_ptrs, mode_idxs,
    size_mode_ptr, size_mode_idx,
    ranks_v, factor_sizes_v, dims_v,
    /*ncm=*/0, /*order=*/4, tile.cache);
  return true;
}

static void prepareTTMcTiles3D(
  const CSFCopy& csf,
  uint64_t* ranks_v,
  uint64_t* factor_sizes_v,
  uint64_t* dims_v,
  uint64_t tile_rows,
  std::vector<TTMCTile3D>& tiles)
{
  tiles.clear();
  uint64_t num_roots = (uint64_t)csf.idxs[0].size();
  for (uint64_t begin = 0; begin < num_roots; begin += tile_rows) {
    tiles.emplace_back();
    if (!buildTTMCTile3D(csf, begin, tile_rows, ranks_v, factor_sizes_v, dims_v, tiles.back())) {
      throw std::runtime_error("Failed to build tiled CSF view.");
    }
  }
}

static void prepareTTMcTiles4D(
  const CSFCopy& csf,
  uint64_t* ranks_v,
  uint64_t* factor_sizes_v,
  uint64_t* dims_v,
  uint64_t tile_rows,
  std::vector<TTMCTile3D>& tiles)
{
  tiles.clear();
  uint64_t num_roots = (uint64_t)csf.idxs[0].size();
  for (uint64_t begin = 0; begin < num_roots; begin += tile_rows) {
    tiles.emplace_back();
    if (!buildTTMCTile4D(csf, begin, tile_rows, ranks_v, factor_sizes_v, dims_v, tiles.back())) {
      throw std::runtime_error("Failed to build tiled 4D CSF view.");
    }
  }
}

static void freeTTMcTiles3D(std::vector<TTMCTile3D>& tiles) {
  for (auto& tile : tiles) freeTTMCTile3D(tile);
  tiles.clear();
}

#if !SCALAR_DOUBLE
static void accumulateGramFromTileFP32(
  cublasHandle_t cublasH,
  const TiledModeWorkspace& ws,
  double* d_gram_dp,
  uint64_t tile_rows,
  int cols,
  bool& gram_initialized)
{
  const double alpha = 1.0;
  const bool use_syrk = useSyrkForGram(cols);
  for (uint64_t start = 0; start < tile_rows; start += (uint64_t)ws.dp_block_rows) {
    int block_rows = (int)std::min<uint64_t>((uint64_t)ws.dp_block_rows, tile_rows - start);
    long long elems = (long long)block_rows * cols;
    convert_scalar_to_dp<<<(elems + 255) / 256, 256>>>(
      ws.d_y_tile + start * (uint64_t)cols,
      ws.d_tile_block_dp,
      elems);
    CHECK_CUDA(cudaGetLastError());
    double beta = gram_initialized ? 1.0 : 0.0;
    if (use_syrk) {
      CHECK_CUBLAS(cublasDsyrk(
        cublasH,
        CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        cols, block_rows,
        &alpha,
        ws.d_tile_block_dp, cols,
        &beta,
        d_gram_dp, cols));
    } else {
      CHECK_CUBLAS(cublasDgemm(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_T,
        cols, cols, block_rows,
        &alpha,
        ws.d_tile_block_dp, cols,
        ws.d_tile_block_dp, cols,
        &beta,
        d_gram_dp, cols));
    }
    gram_initialized = true;
  }
}
#endif

static void accumulateGramFromTile(
  cublasHandle_t cublasH,
  const TiledModeWorkspace& ws,
#if SCALAR_DOUBLE
  scalar_t* d_gram,
#else
  double* d_gram,
#endif
  uint64_t tile_rows,
  int cols,
  bool& gram_initialized)
{
#if SCALAR_DOUBLE
  scalar_t alpha = (scalar_t)1;
  scalar_t beta = gram_initialized ? (scalar_t)1 : (scalar_t)0;
  if (useSyrkForGram(cols)) {
    CHECK_CUBLAS(cublasSyrkT(
      cublasH,
      CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
      cols, (int)tile_rows,
      &alpha,
      ws.d_y_tile, cols,
      &beta,
      d_gram, cols));
  } else {
    CHECK_CUBLAS(cublasGemmT(
      cublasH,
      CUBLAS_OP_N, CUBLAS_OP_T,
      cols, cols, (int)tile_rows,
      &alpha,
      ws.d_y_tile, cols,
      ws.d_y_tile, cols,
      &beta,
      d_gram, cols));
  }
  gram_initialized = true;
#else
  accumulateGramFromTileFP32(cublasH, ws, d_gram, tile_rows, cols, gram_initialized);
#endif
}

static bool solveTallGramEigensystem(
  cusolverDnHandle_t cusolverH,
  cublasHandle_t cublasH,
#if SCALAR_DOUBLE
  scalar_t* d_gram,
#else
  double* d_gram,
#endif
  int M_total,
  int cols,
  int rank,
  TiledGramSVDResult& result,
  bool* resolution_failed_out,
  bool verbose)
{
  if (resolution_failed_out) *resolution_failed_out = false;
  rank = std::min(rank, cols);
  result.cols = cols;
  result.rank = rank;

#if SCALAR_DOUBLE
  scalar_t traceATA = 0;
  CHECK_CUBLAS(cublasDasumT(cublasH, cols, d_gram, cols + 1, &traceATA));

  scalar_t* d_W = nullptr;
  scalar_t* d_work = nullptr;
  int* d_info = nullptr;
  int lwork = 0;
  CHECK_CUDA(cudaMalloc(&d_W, sizeof(scalar_t) * cols));
  CHECK_CUSOLVER(cusolverSyevdBufSizeT(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
    cols, d_gram, cols, d_W, &lwork));
  CHECK_CUDA(cudaMalloc(&d_work, sizeof(scalar_t) * lwork));
  CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
  CHECK_CUSOLVER(cusolverSyevdT(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
    cols, d_gram, cols, d_W, d_work, lwork, d_info));

  int h_info = 0;
  CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info != 0) {
    std::cerr << "  WARNING: tiled Gram eigendecomposition failed with info=" << h_info << "\n";
    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(d_info));
    return false;
  }

  CHECK_CUDA(cudaMemset(d_info, 0, sizeof(int)));
  if (rank > 1)
    check_resolution<<<1, rank - 1>>>(d_W, M_total + cols, cols, rank, eps_mach, d_info, traceATA);
  CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info != 0) {
    if (verbose)
      std::cout << "  tiled Gram resolution check failed; full fallback is unavailable in tiled mode\n";
    if (resolution_failed_out) *resolution_failed_out = true;
    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(d_info));
    return false;
  }

  std::vector<scalar_t> h_top(rank);
  CHECK_CUDA(cudaMemcpy(
    h_top.data(),
    d_W + (cols - rank),
    sizeof(scalar_t) * rank,
    cudaMemcpyDeviceToHost));
  for (scalar_t lam : h_top) result.core_norm_sq += std::max((double)lam, 0.0);

  CHECK_CUDA(cudaMalloc(&result.d_VR, sizeof(scalar_t) * cols * rank));
  CHECK_CUDA(cudaMemcpy(
    result.d_VR,
    d_gram + (long long)(cols - rank) * cols,
    sizeof(scalar_t) * cols * rank,
    cudaMemcpyDeviceToDevice));

  CHECK_CUDA(cudaMalloc(&result.d_inv_sigma, sizeof(scalar_t) * rank));
  compute_inv_sigma<<<(rank + 255) / 256, 256>>>(d_W, result.d_inv_sigma, cols, rank, (scalar_t)1e-12);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaFree(d_W));
  CHECK_CUDA(cudaFree(d_work));
  CHECK_CUDA(cudaFree(d_info));
#else
  const double eps_mach_dp = 1.1102230246251565e-16;
  double traceATA = 0;
  CHECK_CUBLAS(cublasDasum(cublasH, cols, d_gram, cols + 1, &traceATA));

  double* d_W = nullptr;
  double* d_work = nullptr;
  int* d_info = nullptr;
  int lwork = 0;
  CHECK_CUDA(cudaMalloc(&d_W, sizeof(double) * cols));
  bool used_topr_solver = forceGpuIterativeEig();
  cusolverStatus_t full_st = CUSOLVER_STATUS_SUCCESS;
  if (!used_topr_solver) {
    full_st = cusolverDnDsyevd_bufferSize(
      cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
      cols, d_gram, cols, d_W, &lwork);
    if (full_st != CUSOLVER_STATUS_SUCCESS) {
      if (verbose) {
        std::cout << "  tiled full eigensolver bufferSize failed with status="
                  << cusolverStatusString(full_st)
                  << ", trying exact top-R eigensolver\n";
      }
      used_topr_solver = true;
    } else {
      cudaError_t work_err = cudaMalloc(&d_work, sizeof(double) * std::max(lwork, 1));
      if (work_err != cudaSuccess) {
        cudaGetLastError();
        used_topr_solver = true;
        if (verbose) {
          std::cout << "  tiled full eigensolver work alloc failed, trying exact top-R eigensolver\n";
        }
      } else {
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
        full_st = cusolverDnDsyevd(
          cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
          cols, d_gram, cols, d_W, d_work, lwork, d_info);
        if (full_st != CUSOLVER_STATUS_SUCCESS) {
          if (verbose) {
            std::cout << "  tiled full eigensolver failed with status="
                      << cusolverStatusString(full_st)
                      << ", trying exact top-R eigensolver\n";
          }
          CHECK_CUDA(cudaFree(d_work));
          d_work = nullptr;
          CHECK_CUDA(cudaFree(d_info));
          d_info = nullptr;
          used_topr_solver = true;
        }
      }
    }
  } else if (verbose) {
    if (forceGpuIterativeEig()) {
      std::cout << "  exact tiled top-R eigensolver forced to iterative GPU fallback\n";
    }
  }

  if (!used_topr_solver) {
    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
      std::cerr << "  WARNING: tiled Gram eigendecomposition failed with info=" << h_info << "\n";
      CHECK_CUDA(cudaFree(d_W));
      CHECK_CUDA(cudaFree(d_work));
      CHECK_CUDA(cudaFree(d_info));
      return false;
    }

    CHECK_CUDA(cudaMemset(d_info, 0, sizeof(int)));
    if (rank > 1)
      check_resolution_dp<<<1, rank - 1>>>(d_W, M_total + cols, cols, rank, eps_mach_dp, d_info, traceATA);
     CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
     if (h_info != 0) {
       if (verbose)
         std::cout << "  tiled Gram resolution check failed; full fallback is unavailable in tiled mode\n";
       if (resolution_failed_out) *resolution_failed_out = true;
       CHECK_CUDA(cudaFree(d_W));
       CHECK_CUDA(cudaFree(d_work));
       CHECK_CUDA(cudaFree(d_info));
       return false;
    }

    std::vector<double> h_top(rank);
    CHECK_CUDA(cudaMemcpy(
      h_top.data(),
      d_W + (cols - rank),
      sizeof(double) * rank,
      cudaMemcpyDeviceToHost));
    for (double lam : h_top) result.core_norm_sq += std::max(lam, 0.0);

    CHECK_CUDA(cudaMalloc(&result.d_VR, sizeof(scalar_t) * cols * rank));
    convert_dp_to_scalar<<<((long long)cols * rank + 255) / 256, 256>>>(
      d_gram + (long long)(cols - rank) * cols,
      result.d_VR,
      (long long)cols * rank);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMalloc(&result.d_inv_sigma, sizeof(scalar_t) * rank));
    compute_inv_sigma_dp<<<(rank + 255) / 256, 256>>>(d_W, result.d_inv_sigma, cols, rank, 1e-12);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(d_info));
  } else {
    CHECK_CUDA(cudaFree(d_W));
    d_W = nullptr;
    bool top_r_ok = false;
    bool top_r_resolution_failed = false;
    if (forceGpuIterativeEig()) {
      top_r_ok = solveTopREigensystemDpIterativeGpu(
        cusolverH, cublasH, d_gram, cols, rank, M_total + cols, traceATA,
        &result.d_VR, &result.d_inv_sigma, &result.core_norm_sq,
        &top_r_resolution_failed, verbose);
    } else {
      top_r_ok = solveTopREigensystemDpExact(
        cusolverH, d_gram, cols, rank, M_total + cols, traceATA,
        &result.d_VR, &result.d_inv_sigma, &result.core_norm_sq,
        &top_r_resolution_failed, verbose);
      if (!top_r_ok && !top_r_resolution_failed && allowGpuIterativeEigFallback()) {
        if (verbose) {
          std::cout << "  exact GPU top-R eigensolver failed in tiled path, trying iterative GPU fallback\n";
        }
        bool iterative_resolution_failed = false;
        top_r_ok = solveTopREigensystemDpIterativeGpu(
          cusolverH, cublasH, d_gram, cols, rank, M_total + cols, traceATA,
          &result.d_VR, &result.d_inv_sigma, &result.core_norm_sq,
          &iterative_resolution_failed, verbose);
        top_r_resolution_failed = top_r_resolution_failed || iterative_resolution_failed;
      }
    }
    if (!top_r_ok) {
      if (top_r_resolution_failed && resolution_failed_out) {
        *resolution_failed_out = true;
      }
      if (d_work) CHECK_CUDA(cudaFree(d_work));
      if (d_info) CHECK_CUDA(cudaFree(d_info));
      return false;
    }
  }
#endif

  return true;
}

static TiledModeUpdateStats runTiledModeUpdate(
  cusolverDnHandle_t cusolverH,
  cublasHandle_t cublasH,
  std::vector<TTMCTile3D>& tiles,
  const TiledModeWorkspace& ws,
  scalar_t** d_factor_mats_v,
  scalar_t* d_factor_out,
  uint64_t factor_size_out,
  uint64_t* ranks_v,
  int order,
  int M_total,
  bool split_workspace_for_tiled_4d,
  bool verbose)
{
  TiledModeUpdateStats stats;
  int cols = 1;
  for (int l = 1; l < order; ++l) cols *= (int)ranks_v[l];
  const int rank = (int)ranks_v[0];
  scalar_t alpha = (scalar_t)1;
  scalar_t beta = (scalar_t)0;
  uint64_t max_rows = 0;
  for (const auto& tile : tiles) max_rows = std::max(max_rows, tile.num_rows);

  auto total_start = std::chrono::high_resolution_clock::now();

  const TiledModeWorkspace* pass1_ws = &ws;
  TiledModeWorkspace local_pass1_ws;
  TiledModeWorkspace local_pass2_ws;
  if (split_workspace_for_tiled_4d) {
    if (!tryPrepareTiledModeWorkspace(
          local_pass1_ws,
          tiledWorkspaceYElems(max_rows, cols),
          /*factor_capacity_elems=*/0,
#if !SCALAR_DOUBLE
          tiledWorkspaceDPElems(max_rows, cols)
#endif
        )) {
      throw std::runtime_error("Failed to allocate tiled pass-1 workspace.");
    }
    pass1_ws = &local_pass1_ws;
  }

#if SCALAR_DOUBLE
  scalar_t* d_gram = nullptr;
  CHECK_CUDA(cudaMalloc(&d_gram, sizeof(scalar_t) * cols * cols));
#else
  double* d_gram = nullptr;
  CHECK_CUDA(cudaMalloc(&d_gram, sizeof(double) * cols * cols));
#endif
  bool gram_initialized = false;

  for (auto& tile : tiles) {
    auto t0 = std::chrono::high_resolution_clock::now();
    run_ttmc_cuda(
      tile.d_ptrs.data(), tile.d_idxs.data(), tile.d_values,
      d_factor_mats_v,
      pass1_ws->d_y_tile, tile.output_elems,
      /*ncm=*/0, ranks_v, order,
      tile.cache, /*log_method=*/false);
    auto t1 = std::chrono::high_resolution_clock::now();
    stats.ttmc_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    accumulateGramFromTile(cublasH, *pass1_ws, d_gram, tile.num_rows, cols, gram_initialized);
  }

  if (split_workspace_for_tiled_4d) {
    freeTiledModeWorkspace(local_pass1_ws);
  }

  TiledGramSVDResult svd_result;
  bool tiled_resolution_failed = false;
  if (!solveTallGramEigensystem(
        cusolverH, cublasH, d_gram, M_total, cols, rank, svd_result,
        &tiled_resolution_failed, verbose)) {
#if SCALAR_DOUBLE
    CHECK_CUDA(cudaFree(d_gram));
#else
    CHECK_CUDA(cudaFree(d_gram));
#endif
    if (tiled_resolution_failed) {
      throw TiledGramResolutionFailed();
    }
    throw std::runtime_error("Tiled Gram eigensolve failed; full fallback is unavailable in tiled mode.");
  }

  CHECK_CUDA(cudaFree(d_gram));
  CHECK_CUDA(cudaMemset(d_factor_out, 0, sizeof(scalar_t) * factor_size_out));

  const TiledModeWorkspace* pass2_ws = &ws;
  if (split_workspace_for_tiled_4d) {
    if (!tryPrepareTiledModeWorkspace(
          local_pass2_ws,
          tiledWorkspaceYElems(max_rows, cols),
          tiledWorkspaceFactorElems(max_rows, rank),
#if !SCALAR_DOUBLE
          /*dp_capacity_elems=*/0
#endif
        )) {
      freeTiledGramSVDResult(svd_result);
      throw std::runtime_error("Failed to allocate tiled pass-2 workspace.");
    }
    pass2_ws = &local_pass2_ws;
  }

  for (auto& tile : tiles) {
    auto t0 = std::chrono::high_resolution_clock::now();
    run_ttmc_cuda(
      tile.d_ptrs.data(), tile.d_idxs.data(), tile.d_values,
      d_factor_mats_v,
      pass2_ws->d_y_tile, tile.output_elems,
      /*ncm=*/0, ranks_v, order,
      tile.cache, /*log_method=*/false);
    auto t1 = std::chrono::high_resolution_clock::now();
    stats.ttmc_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    CHECK_CUBLAS(cublasGemmT(
      cublasH,
      CUBLAS_OP_T, CUBLAS_OP_N,
      rank, (int)tile.num_rows, cols,
      &alpha,
      svd_result.d_VR, cols,
      pass2_ws->d_y_tile, cols,
      &beta,
      pass2_ws->d_factor_tile, rank));
    CHECK_CUBLAS(cublasDgmmT(
      cublasH,
      CUBLAS_SIDE_LEFT,
      rank, (int)tile.num_rows,
      pass2_ws->d_factor_tile, rank,
      svd_result.d_inv_sigma, 1,
      pass2_ws->d_factor_tile, rank));

    uint64_t total_scatter = tile.num_rows * (uint64_t)rank;
    scatter_factor_rows_kernel<<<(total_scatter + 255) / 256, 256>>>(
      tile.d_global_rows, pass2_ws->d_factor_tile, d_factor_out, tile.num_rows, rank);
    CHECK_CUDA(cudaGetLastError());
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  auto total_end = std::chrono::high_resolution_clock::now();
  stats.total_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
  stats.core_norm_sq = svd_result.core_norm_sq;
  if (split_workspace_for_tiled_4d) {
    freeTiledModeWorkspace(local_pass2_ws);
  }
  freeTiledGramSVDResult(svd_result);
  return stats;
}


// ===================================================================
// CPU reference TTMc (ncm_0 logic from v2_cpu_factorize_n_fuse.cu)
// Mirrors GPU_4L_CM_device_func_ncm_0 exactly for verification.
 // arr_A: row-major (I_level1, f1), arr_B: row-major (I_level2, f2)
 // arr_O: row-major (I_root, f1, f2) — must be zeroed by caller
 // ===================================================================
 void cpu_ttmc_ncm0(const CSFCopy& csf,
                    const scalar_t* arr_A, const scalar_t* arr_B,
                    scalar_t* arr_O, uint64_t /*I_root*/,
                    uint32_t f1, uint32_t f2) {
   std::vector<scalar_t> buffer(f2, (scalar_t)0);
   size_t num_roots = csf.idxs[0].size();
 
   for (size_t i_ptr = 0; i_ptr < num_roots; ++i_ptr) {
     uint64_t i = csf.idxs[0][i_ptr];
 
     for (uint64_t j_ptr = csf.ptrs[1][i_ptr]; j_ptr < csf.ptrs[1][i_ptr + 1]; ++j_ptr) {
       uint64_t j = csf.idxs[1][j_ptr];
 
       std::fill(buffer.begin(), buffer.end(), (scalar_t)0);
 
       for (uint64_t k_ptr = csf.ptrs[2][j_ptr]; k_ptr < csf.ptrs[2][j_ptr + 1]; ++k_ptr) {
         uint64_t k = csf.idxs[2][k_ptr];
         scalar_t val = csf.values[k_ptr];
         for (uint32_t s = 0; s < f2; ++s)
           buffer[s] += val * arr_B[k * f2 + s];
       }
 
       for (uint32_t r = 0; r < f1; ++r) {
         scalar_t a_jr = arr_A[j * f1 + r];
         for (uint32_t s = 0; s < f2; ++s)
           arr_O[i * f1 * f2 + r * f2 + s] += buffer[s] * a_jr;
       }
     }
   }
 }
 
 // Compare GPU result (arr_O_gpu, already on host) vs CPU result for mode n.
 // Reports max absolute error and relative error.
 void verify_ttmc(const CSFCopy& csf, const scalar_t* arr_A, const scalar_t* arr_B,
                  const scalar_t* arr_O_gpu, uint64_t I_root, uint32_t f1, uint32_t f2, int mode) {
   uint64_t size = I_root * f1 * f2;
   std::vector<scalar_t> arr_O_cpu(size, (scalar_t)0);
   cpu_ttmc_ncm0(csf, arr_A, arr_B, arr_O_cpu.data(), I_root, f1, f2);
 
   scalar_t max_err = (scalar_t)0, max_val = (scalar_t)0;
   for (uint64_t k = 0; k < size; k++) {
     scalar_t diff = std::fabs(arr_O_gpu[k] - arr_O_cpu[k]);
     if (diff > max_err) max_err = diff;
     scalar_t v = std::fabs(arr_O_cpu[k]);
     if (v > max_val) max_val = v;
   }
   scalar_t rel_err = (max_val > (scalar_t)0) ? max_err / max_val : max_err;
   std::cout << "[check] Mode-" << mode << " TTMc: max_abs_err=" << max_err
             << "  rel_err=" << rel_err
             << "  max_cpu_val=" << max_val << "\n";
 }
 
 
 int main(int argc, char* argv[]) {
   bool verbose = false;
   bool check   = false;
   g_gpu_stats  = false;
  std::string tns_file;
  std::vector<uint64_t> ranks;
  int max_iters = 25;
  scalar_t tol = (scalar_t)1e-5;
  TTMcStorageMode ttmc_path = TTMcStorageMode::Auto;
  uint64_t tile_roots_override = 0;
  int tile_mb_hint = 0;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-v" || arg == "--verbose") verbose = true;
    else if (arg == "-c" || arg == "--check") check = true;
    else if (arg == "-g" || arg == "--gpu-stats") g_gpu_stats = true;
    else if (arg == "--ttmc-path" && i + 1 < argc) {
      ttmc_path = parseTTMcStorageMode(argv[++i]);
    } else if (arg == "--tile-roots" && i + 1 < argc) {
      tile_roots_override = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (arg == "--tile-mb" && i + 1 < argc) {
      tile_mb_hint = std::max(1, atoi(argv[++i]));
    }
    else if ((arg == "-r" || arg == "--ranks") && i + 1 < argc) {
      ranks.clear();
      while (i + 1 < argc && argv[i + 1][0] != '-')
        ranks.push_back(static_cast<uint64_t>(atoi(argv[++i])));
     } else if ((arg == "-m" || arg == "--max-iters") && i + 1 < argc) {
       max_iters = atoi(argv[++i]);
     } else if ((arg == "-t" || arg == "--tol") && i + 1 < argc) {
       tol = static_cast<scalar_t>(atof(argv[++i]));
     } else if (tns_file.empty()) {
       tns_file = arg;
     }
   }
 
   if (tns_file.empty()) {
     std::cerr << "Usage: " << argv[0] << " [options] <tns_file>\n"
          << "Options:\n"
          << "  -v, --verbose         Detailed per-step timing and SVD internals\n"
          << "  -c, --check           CPU reference TTMc in iter-0 (correctness check)\n"
          << "  -g, --gpu-stats       Print size of every GPU cudaMalloc allocation\n"
          << "  -r, --ranks R0 R1 ...  Target ranks, one per mode (default 10 per mode)\n"
          << "  -m, --max-iters N     Max HOOI iterations (default 25)\n"
          << "  -t, --tol T           Convergence tolerance on fit (default 1e-5)\n"
          << "  --ttmc-path MODE      auto|full|tiled (default auto)\n"
          << "  --tile-roots N        Override tiled root-slices per tile\n"
         << "  --tile-mb N           Approximate Y-tile buffer size in MiB (default auto)\n";
     return 1;
   }
   #if SCALAR_DOUBLE
   std::cout << "Using double precision for TTMc" << std::endl;
   #else
   std::cout << "Using float precision for TTMc" << std::endl;
   #endif
 
   try {
     // ===================================================================
     // 1. Read COO tensor from .tns file
     // ===================================================================
     COOTensor coo = readCOOTensor(tns_file);
     int order = coo.order;
     if (order < 3 || order > 4) {
       std::cerr << "Error: Tucker HOOI supports 3D and 4D tensors only (got order "
                 << order << ")\n";
       return 1;
     }
     while ((int)ranks.size() < order) ranks.push_back(10);
     for (int i = 0; i < order; i++) ranks[i] = std::min(ranks[i], coo.dims[i]);

     const TensorOptimizationPolicy opt_policy =
       chooseTensorOptimizationPolicy(tns_file, ranks);
     g_enable_pinned_blocked_full_svd = opt_policy.enable_pinned_blocked_full_svd;
     g_force_gpu_iterative_eig = opt_policy.force_iterative_eig;
     g_allow_gpu_iterative_eig_fallback = opt_policy.allow_iterative_eig_fallback;
 
     std::cout << "Tensor:";
     for (int i = 0; i < order; i++) std::cout << " " << coo.dims[i];
     std::cout << "  nnz=" << coo.values.size() << "  ranks=(";
     for (int i = 0; i < order; i++) { if (i) std::cout << ","; std::cout << ranks[i]; }
     std::cout << ")\n";
     std::cout << "Optimization policy: legacy-first"
               << ", pinned-blocked-full-svd="
               << (g_enable_pinned_blocked_full_svd ? "enabled" : "disabled")
               << ", iterative-eig-fallback="
               << (g_allow_gpu_iterative_eig_fallback ? "enabled" : "disabled")
               << ", forced-iterative-eig="
               << (g_force_gpu_iterative_eig ? "enabled" : "disabled")
               << "\n";
 
     // ===================================================================
     // 2. Unique-index counts per mode (for CSF compression ordering)
     // ===================================================================
     std::vector<size_t> uniqueCounts(order);
     for (int m = 0; m < order; m++) {
       std::unordered_set<uint64_t> uniq;
       for (const auto& idx : coo.indices) uniq.insert(idx[m]);
       uniqueCounts[m] = uniq.size();
     }
     if (verbose) {
       std::cout << "Unique indices per mode:";
       for (int m = 0; m < order; m++) std::cout << " " << uniqueCounts[m];
       std::cout << "\n";
     }
 
     // ===================================================================
     // 3. Build `order` CSF copies (mode n as root → used for TTMc-n)
     // ===================================================================
     std::vector<CSFCopy> csf_copies(order);
     for (int n = 0; n < order; n++) {
       csf_copies[n] = buildCSFCopy(coo, n, uniqueCounts);
       if (verbose) {
         std::cout << "CSF copy " << n << " (root=mode" << n << "): levels=[";
         for (int l = 0; l < order; l++) {
           if (l) std::cout << ",";
           std::cout << csf_copies[n].modeOrder[l];
         }
         std::cout << "]  roots=" << csf_copies[n].idxs[0].size()
                   << "  nnz=" << csf_copies[n].values.size() << "\n";
       }
     }
 
     // ===================================================================
     // 4. Upload all CSF copies to GPU (persistent across iterations)
     // ===================================================================
     for (int n = 0; n < order; n++) uploadCSFToGPU(csf_copies[n]);
 
     // ===================================================================
     // 5. Allocate and initialize factor matrices on CPU (row-major)
     // ===================================================================
     std::vector<scalar_t*> factors(order);
     std::vector<uint64_t> factor_sizes(order);
     const char* factors_dir_env = std::getenv("TUCKER_FACTORS_DIR");
     const char* dump_dir_env = std::getenv("TUCKER_FACTORS_DUMP_DIR");
     for (int i = 0; i < order; i++) {
       factor_sizes[i] = coo.dims[i] * ranks[i];
       factors[i] = new scalar_t[factor_sizes[i]];
       bool loaded = false;
       if (factors_dir_env && *factors_dir_env) {
         const std::string factor_path = factorBinPath(factors_dir_env, i);
         loaded = load_factor_bin(factor_path, factors[i], coo.dims[i], ranks[i]);
         if (loaded) {
           std::cout << "[factors] loaded " << factor_path
                     << " (" << coo.dims[i] << " x " << ranks[i] << ")\n";
         }
         else {
           throw std::runtime_error("missing factor init file: " + factor_path);
         }
       }
       if (!loaded) {
         init_factor_orthonormal(coo.dims[i], ranks[i], 42 + i, factors[i]);
       }
       if (dump_dir_env && *dump_dir_env) {
         const std::string factor_path = factorBinPath(dump_dir_env, i);
         dump_factor_bin(factor_path, factors[i], coo.dims[i], ranks[i]);
         std::cout << "[factors] dumped " << factor_path
                   << " (" << coo.dims[i] << " x " << ranks[i] << ")\n";
       }
     }
 
     cusolverDnHandle_t cusolverH = nullptr;
     cublasHandle_t cublasH = nullptr;
     CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
     CHECK_CUBLAS(cublasCreate(&cublasH));
     CHECK_CUBLAS(cublasSetMathMode(cublasH, CUBLAS_PEDANTIC_MATH));
 
     // Factor matrices on GPU — always kept row-major (same layout as CPU)
     std::vector<scalar_t*> d_factors(order, nullptr);
     for (int i = 0; i < order; i++) {
       CHECK_CUDA(cudaMalloc(&d_factors[i], sizeof(scalar_t) * factor_sizes[i]));
       CHECK_CUDA(cudaMemcpy(d_factors[i], factors[i],
         sizeof(scalar_t) * factor_sizes[i], cudaMemcpyHostToDevice));
     }
 
     // ===================================================================
     // 6. Decide which modes keep full Y and which modes switch to tiled/recompute
     // ===================================================================
     std::vector<uint64_t> arr_O_sizes(order);
     std::vector<size_t> arr_O_bytes(order, 0);
     std::vector<uint64_t> rank_products(order, 1);
     for (int n = 0; n < order; n++) {
       for (int l = 1; l < order; l++)
         rank_products[n] *= ranks[csf_copies[n].modeOrder[l]];
       arr_O_sizes[n] = coo.dims[n] * rank_products[n];
       arr_O_bytes[n] = (size_t)arr_O_sizes[n] * sizeof(scalar_t);
     }

     if (ttmc_path == TTMcStorageMode::Tiled && order != 3 && order != 4) {
       throw std::runtime_error("The tiled/recompute path is currently implemented only for 3D/4D tensors.");
     }

     std::vector<bool> use_tiled(order, false);
     std::vector<bool> supports_tiled(order, false);
     for (int n = 0; n < order; n++) {
       supports_tiled[n] = (order == 3 || order == 4);
       if (!supports_tiled[n]) {
         use_tiled[n] = false;
         continue;
       }

     if (ttmc_path == TTMcStorageMode::Tiled) {
        use_tiled[n] = true;
      } else {
        use_tiled[n] = false;
      }
     }

     auto choose_mode_to_tile = [&]() -> int {
       int best = -1;
       size_t best_bytes = 0;
       for (int n = 1; n < order; n++) {
         if (!supports_tiled[n] || use_tiled[n]) continue;
         if (arr_O_bytes[n] > best_bytes) {
           best_bytes = arr_O_bytes[n];
           best = n;
         }
       }
       if (best >= 0) return best;
       if (supports_tiled[0] && !use_tiled[0]) return 0;
       return -1;
     };

     auto recompute_max_full_arr_O_size = [&]() -> uint64_t {
       uint64_t max_size = 0;
       for (int n = 0; n < order; n++)
         if (!use_tiled[n]) max_size = std::max(max_size, arr_O_sizes[n]);
       return max_size;
     };

     uint64_t max_full_arr_O_size = recompute_max_full_arr_O_size();

     scalar_t* d_arr_O = nullptr;
     while (max_full_arr_O_size > 0) {
       // Legacy-first auto policy: keep every mode on the full-Y path until
       // the actual full-buffer allocation fails, then tile one mode and retry.
       cudaError_t alloc_err = cudaMalloc(&d_arr_O, sizeof(scalar_t) * max_full_arr_O_size);
       if (alloc_err == cudaSuccess) break;
       cudaGetLastError();
       d_arr_O = nullptr;

       if (ttmc_path != TTMcStorageMode::Auto) {
         CHECK_CUDA(alloc_err);
       }

       int fallback_mode = choose_mode_to_tile();
       if (fallback_mode < 0) {
         CHECK_CUDA(alloc_err);
       }

       use_tiled[fallback_mode] = true;
       max_full_arr_O_size = recompute_max_full_arr_O_size();
       std::cout << "Auto path fallback: switching mode " << fallback_mode
                 << " to tiled after full-buffer allocation failed\n";
     }

     std::cout << "TTMc path request: " << ttmcStorageModeName(ttmc_path) << "\n";
     for (int n = 0; n < order; n++) {
       std::cout << "  mode " << n
                 << ": dense Y would be " << formatBytes((double)arr_O_bytes[n])
                 << " -> using " << (use_tiled[n] ? "tiled" : "full") << "\n";
     }

     scalar_t* arr_O_host = nullptr;
     if (check && max_full_arr_O_size > 0) {
       arr_O_host = allocate_aligned_array(max_full_arr_O_size);
     }

     // ===================================================================
     // 7. Prepare TTMc caches / tiled workspaces before the HOOI loop
     // ===================================================================
     std::vector<TTMcCache> ttmc_caches(order);
     std::vector<std::vector<TTMCTile3D>> tiled_mode_tiles(order);
     TiledModeWorkspace tiled_workspace;
     std::vector<uint64_t> tiled_tile_rows(order, 0);
     std::vector<std::vector<uint64_t>> tiled_ranks_v(order);
     std::vector<std::vector<uint64_t>> tiled_dims_v(order);
     std::vector<std::vector<uint64_t>> tiled_fsizes_v(order);
     const size_t tile_reserve_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
     int tiled_modes_remaining = 0;
     for (int n = 0; n < order; n++)
       if (use_tiled[n]) ++tiled_modes_remaining;

     auto rebuild_tiled_mode = [&](int n) {
       freeTTMcTiles3D(tiled_mode_tiles[n]);
       if (order == 3) {
         prepareTTMcTiles3D(
           csf_copies[n],
           tiled_ranks_v[n].data(), tiled_fsizes_v[n].data(), tiled_dims_v[n].data(),
           tiled_tile_rows[n],
           tiled_mode_tiles[n]);
       } else {
         prepareTTMcTiles4D(
           csf_copies[n],
           tiled_ranks_v[n].data(), tiled_fsizes_v[n].data(), tiled_dims_v[n].data(),
           tiled_tile_rows[n],
           tiled_mode_tiles[n]);
       }
     };

     for (int n = 0; n < order; n++) {
       CSFCopy& csf = csf_copies[n];
       std::vector<uint64_t*> ptrs_raw(order), idxs_raw(order);
       std::vector<uint64_t>  size_ptr(order), size_idx(order);
       std::vector<uint64_t>  ranks_v(order), dims_v(order), fsizes_v(order, 0);
       for (int l = 0; l < order; l++) {
         ptrs_raw[l] = csf.ptrs[l].data();
         idxs_raw[l] = csf.idxs[l].data();
         size_ptr[l] = (uint64_t)csf.ptrs[l].size();
         size_idx[l] = (uint64_t)csf.idxs[l].size();
         dims_v[l]   = coo.dims[csf.modeOrder[l]];
         ranks_v[l]  = (l == 0) ? ranks[n] : ranks[csf.modeOrder[l]];
         if (l > 0) fsizes_v[l] = factor_sizes[csf.modeOrder[l]];
       }

       if (!use_tiled[n]) {
         prepare_ttmc_cuda(ptrs_raw.data(), idxs_raw.data(),
                           size_ptr.data(), size_idx.data(),
                           ranks_v.data(), fsizes_v.data(), dims_v.data(),
                           /*ncm=*/0, order, ttmc_caches[n]);
       } else {
         tiled_ranks_v[n] = ranks_v;
         tiled_dims_v[n] = dims_v;
         tiled_fsizes_v[n] = fsizes_v;
         size_t tile_target_bytes = 0;
         if (tile_mb_hint > 0) {
           tile_target_bytes = (size_t)tile_mb_hint * 1024ULL * 1024ULL;
         } else {
           size_t free_bytes = 0, total_bytes = 0;
           CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
           size_t usable_bytes = (free_bytes > tile_reserve_bytes)
             ? (free_bytes - tile_reserve_bytes)
             : std::max<size_t>(free_bytes / 2, 1ULL << 20);
           tile_target_bytes = std::max<size_t>(
             usable_bytes / std::max(tiled_modes_remaining, 1),
             (size_t)(rank_products[n] + ranks[n]) * sizeof(scalar_t));
         }
         uint64_t tile_rows = chooseTileRootCount(
           (uint64_t)csf.idxs[0].size(),
           rank_products[n],
           ranks[n],
           tile_roots_override,
           tile_target_bytes);
         tiled_tile_rows[n] = tile_rows;
         rebuild_tiled_mode(n);
         std::cout << "  mode " << n << ": tiled into " << tiled_mode_tiles[n].size()
                   << " tiles of up to " << tile_rows << " root slices"
                   << " (tile budget " << formatBytes((double)tile_target_bytes) << ")\n";
         --tiled_modes_remaining;
       }
     }

    auto choose_tiled_mode_to_shrink = [&]() -> int {
      int best = -1;
      size_t best_bytes = 0;
      for (int n = 0; n < order; n++) {
        if (!use_tiled[n] || tiled_tile_rows[n] <= 1) continue;
        size_t bytes = tiledModeWorkspaceBytes(tiled_tile_rows[n], rank_products[n], ranks[n]);
         if (bytes > best_bytes) {
           best_bytes = bytes;
           best = n;
         }
      }
      return best;
    };

    bool need_shared_tiled_workspace_in_loop = false;
    for (int n = 0; n < order; n++) {
      if (!use_tiled[n]) continue;
      if (order != 4) {
        need_shared_tiled_workspace_in_loop = true;
        break;
      }
    }

    if (need_shared_tiled_workspace_in_loop) {
      while (true) {
        uint64_t max_y_elems = 0;
        uint64_t max_factor_elems = 0;
#if !SCALAR_DOUBLE
        uint64_t max_dp_elems = 0;
#endif
         for (int n = 0; n < order; n++) {
           if (!use_tiled[n]) continue;
           max_y_elems = std::max(
             max_y_elems,
             tiledWorkspaceYElems(tiled_tile_rows[n], (int)rank_products[n]));
           max_factor_elems = std::max(
             max_factor_elems,
             tiledWorkspaceFactorElems(tiled_tile_rows[n], (int)ranks[n]));
#if !SCALAR_DOUBLE
           max_dp_elems = std::max(
             max_dp_elems,
             tiledWorkspaceDPElems(tiled_tile_rows[n], (int)rank_products[n]));
#endif
         }

         bool workspace_ready = tryPrepareTiledModeWorkspace(
           tiled_workspace,
           max_y_elems,
           max_factor_elems,
#if !SCALAR_DOUBLE
           max_dp_elems
#endif
         );
         if (workspace_ready) break;

         int shrink_mode = choose_tiled_mode_to_shrink();
         if (shrink_mode < 0) {
           std::cerr << "Unable to allocate shared tiled workspace even after exhausting tile backoff.\n";
           std::exit(EXIT_FAILURE);
         }

         uint64_t old_rows = tiled_tile_rows[shrink_mode];
         uint64_t new_rows = std::max<uint64_t>(1, old_rows / 2);
         if (new_rows == old_rows && old_rows > 1) --new_rows;
         if (new_rows == 0 || new_rows == old_rows) {
           std::cerr << "Unable to back off tiled mode " << shrink_mode << " any further.\n";
           std::exit(EXIT_FAILURE);
         }

         tiled_tile_rows[shrink_mode] = new_rows;
         rebuild_tiled_mode(shrink_mode);
         std::cout << "  mode " << shrink_mode
                   << ": shared tiled workspace allocation failed, retrying with "
                    << new_rows << " root slices per tile -> "
                    << tiled_mode_tiles[shrink_mode].size() << " tiles\n";
      }
    }

    if (!need_shared_tiled_workspace_in_loop) {
      freeTiledModeWorkspace(tiled_workspace);
    }

    auto needs_tiled_4d_pass1_gram_probe = [&](int n) -> bool {
      if (!use_tiled[n] || order != 4) return false;
      const uint64_t gram_bytes = sizeof(double) * rank_products[n] * rank_products[n];
      const uint64_t hard_4d_gram_threshold = 8ULL * 1024ULL * 1024ULL * 1024ULL;
      return gram_bytes >= hard_4d_gram_threshold;
    };

    for (int n = 0; n < order; n++) {
      if (!needs_tiled_4d_pass1_gram_probe(n)) continue;
      while (!tryAllocateTiledPass1AndGram(tiled_tile_rows[n], rank_products[n])) {
        uint64_t old_rows = tiled_tile_rows[n];
        uint64_t new_rows = std::max<uint64_t>(1, old_rows / 2);
        if (new_rows == old_rows && old_rows > 1) --new_rows;
        if (new_rows == 0 || new_rows == old_rows) {
           throw std::runtime_error("Unable to size tiled pass-1 workspace and Gram together.");
         }
         tiled_tile_rows[n] = new_rows;
         rebuild_tiled_mode(n);
         std::cout << "  mode " << n
                   << ": pass-1 workspace + Gram probe failed, retrying with "
                   << new_rows << " root slices per tile -> "
                   << tiled_mode_tiles[n].size() << " tiles\n";
       }
     }

     // ===================================================================
     // 8. HOOI loop
     // ===================================================================
     scalar_t prev_fit = (scalar_t)0;
     int iter;
     std::vector<double> ttmc_time_us(order, 0.0);
     std::vector<double> svd_time_us(order, 0.0);
     std::vector<bool> late_runtime_tiled_disabled(order, false);
     scalar_t input_tsr_norm = std::sqrt(
       frobenius_norm_sq_sparse(coo.values.data(), coo.values.size()));

     std::cout << "Input tensor ||X||_F = " << input_tsr_norm << "\n";
     std::cout << "Starting HOOI (max_iters=" << max_iters << ", tol=" << tol << ")\n\n";

     auto total_start = std::chrono::high_resolution_clock::now();

     uint64_t I0     = coo.dims[0];
     uint64_t R0     = ranks[0];
     uint64_t N_rest = 1;
     for (int l = 1; l < order; l++) N_rest *= ranks[csf_copies[0].modeOrder[l]];
     scalar_t* d_G_core = nullptr;
     if (!use_tiled[0]) {
       CHECK_CUDA(cudaMalloc(&d_G_core, sizeof(scalar_t) * R0 * N_rest));
     }

     auto ensure_full_arr_O = [&]() {
       if (max_full_arr_O_size == 0 || d_arr_O) return;
       CHECK_CUDA(cudaMalloc(&d_arr_O, sizeof(scalar_t) * max_full_arr_O_size));
     };

     auto ensure_mode0_core_buffer = [&]() {
       if (use_tiled[0] || d_G_core) return;
       CHECK_CUDA(cudaMalloc(&d_G_core, sizeof(scalar_t) * R0 * N_rest));
     };

     auto prepare_full_mode_cache = [&](int n, const std::vector<uint64_t>& ranks_v) {
       if (ttmc_caches[n].initialized) return;
       CSFCopy& cache_csf = csf_copies[n];
       std::vector<uint64_t*> ptrs_raw(order), idxs_raw(order);
       std::vector<uint64_t>  size_ptr(order), size_idx(order);
       std::vector<uint64_t>  dims_v(order), fsizes_v(order, 0);
       for (int l = 0; l < order; l++) {
         ptrs_raw[l] = cache_csf.ptrs[l].data();
         idxs_raw[l] = cache_csf.idxs[l].data();
         size_ptr[l] = (uint64_t)cache_csf.ptrs[l].size();
         size_idx[l] = (uint64_t)cache_csf.idxs[l].size();
         dims_v[l]   = coo.dims[cache_csf.modeOrder[l]];
         if (l > 0) fsizes_v[l] = factor_sizes[cache_csf.modeOrder[l]];
       }
       prepare_ttmc_cuda(
         ptrs_raw.data(), idxs_raw.data(),
         size_ptr.data(), size_idx.data(),
         const_cast<uint64_t*>(ranks_v.data()), fsizes_v.data(), dims_v.data(),
         /*ncm=*/0, order, ttmc_caches[n]);
     };

     auto release_full_mode_state_for_tiled_4d = [&](int current_mode) {
       if (d_arr_O) {
         CHECK_CUDA(cudaFree(d_arr_O));
         d_arr_O = nullptr;
       }
       if (d_G_core) {
         CHECK_CUDA(cudaFree(d_G_core));
         d_G_core = nullptr;
       }
       for (int m = 0; m < order; m++) {
         if (m == current_mode || use_tiled[m]) continue;
         if (ttmc_caches[m].initialized) free_ttmc_cache(ttmc_caches[m]);
       }
     };

     auto tryPromoteModeToTiledAtRuntime = [&](int n, const std::vector<uint64_t>& ranks_v) -> bool {
       if (!supports_tiled[n] || use_tiled[n]) return false;

       std::vector<TTMCTile3D> saved_tiles = std::move(tiled_mode_tiles[n]);
       std::vector<uint64_t> saved_ranks = tiled_ranks_v[n];
       std::vector<uint64_t> saved_dims = tiled_dims_v[n];
       std::vector<uint64_t> saved_fsizes = tiled_fsizes_v[n];
       uint64_t saved_tile_rows = tiled_tile_rows[n];
       bool saved_use_tiled = use_tiled[n];

       auto restore_mode = [&]() {
         freeTTMcTiles3D(tiled_mode_tiles[n]);
         tiled_mode_tiles[n] = std::move(saved_tiles);
         tiled_ranks_v[n] = std::move(saved_ranks);
         tiled_dims_v[n] = std::move(saved_dims);
         tiled_fsizes_v[n] = std::move(saved_fsizes);
         tiled_tile_rows[n] = saved_tile_rows;
         use_tiled[n] = saved_use_tiled;
         max_full_arr_O_size = recompute_max_full_arr_O_size();
       };

       use_tiled[n] = true;
       max_full_arr_O_size = recompute_max_full_arr_O_size();
       tiled_ranks_v[n] = ranks_v;
       tiled_dims_v[n].resize(order);
       tiled_fsizes_v[n].assign(order, 0);
       for (int l = 0; l < order; ++l) {
         tiled_dims_v[n][l] = coo.dims[csf_copies[n].modeOrder[l]];
         if (l > 0) tiled_fsizes_v[n][l] = factor_sizes[csf_copies[n].modeOrder[l]];
       }

       size_t tile_target_bytes = 0;
       if (tile_mb_hint > 0) {
         tile_target_bytes = (size_t)tile_mb_hint * 1024ULL * 1024ULL;
       } else {
         size_t free_bytes = 0, total_bytes = 0;
         CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
         size_t usable_bytes = (free_bytes > tile_reserve_bytes)
           ? (free_bytes - tile_reserve_bytes)
           : std::max<size_t>(free_bytes / 2, 1ULL << 20);
         int active_tiled_modes = 0;
         for (int m = 0; m < order; ++m)
           if (use_tiled[m]) ++active_tiled_modes;
         tile_target_bytes = std::max<size_t>(
           usable_bytes / std::max(active_tiled_modes, 1),
           (size_t)(rank_products[n] + ranks[n]) * sizeof(scalar_t));
       }

       uint64_t tile_rows = chooseTileRootCount(
         (uint64_t)csf_copies[n].idxs[0].size(),
         rank_products[n],
         ranks[n],
         tile_roots_override,
         tile_target_bytes);
       tiled_tile_rows[n] = tile_rows;
       rebuild_tiled_mode(n);
       std::cout << "  mode " << n << ": late tiled fallback starts with "
                 << tiled_mode_tiles[n].size() << " tiles of up to " << tile_rows
                 << " root slices (tile budget " << formatBytes((double)tile_target_bytes) << ")\n";

       if (order != 4) {
         TiledModeWorkspace trial_ws;
         while (true) {
           uint64_t max_y_elems = 0;
           uint64_t max_factor_elems = 0;
#if !SCALAR_DOUBLE
           uint64_t max_dp_elems = 0;
#endif
           for (int m = 0; m < order; ++m) {
             if (!use_tiled[m]) continue;
             max_y_elems = std::max(
               max_y_elems,
               tiledWorkspaceYElems(tiled_tile_rows[m], rank_products[m]));
             max_factor_elems = std::max(
               max_factor_elems,
               tiledWorkspaceFactorElems(tiled_tile_rows[m], ranks[m]));
#if !SCALAR_DOUBLE
             max_dp_elems = std::max(
               max_dp_elems,
               tiledWorkspaceDPElems(tiled_tile_rows[m], rank_products[m]));
#endif
           }

           if (tryPrepareTiledModeWorkspace(
                 trial_ws,
                 max_y_elems,
                 max_factor_elems,
#if !SCALAR_DOUBLE
                 max_dp_elems
#endif
               )) {
             freeTiledModeWorkspace(tiled_workspace);
             tiled_workspace = trial_ws;
             trial_ws = TiledModeWorkspace{};
             break;
           }

           uint64_t old_rows = tiled_tile_rows[n];
           uint64_t new_rows = std::max<uint64_t>(1, old_rows / 2);
           if (new_rows == old_rows && old_rows > 1) --new_rows;
           if (new_rows == 0 || new_rows == old_rows) {
             freeTiledModeWorkspace(trial_ws);
             restore_mode();
             return false;
           }
           tiled_tile_rows[n] = new_rows;
           rebuild_tiled_mode(n);
           std::cout << "  mode " << n
                     << ": late tiled shared-workspace allocation failed, retrying with "
                     << new_rows << " root slices per tile -> "
                     << tiled_mode_tiles[n].size() << " tiles\n";
         }
       }

       if (order == 4) {
         const uint64_t gram_bytes = sizeof(double) * rank_products[n] * rank_products[n];
         const uint64_t hard_4d_gram_threshold = 8ULL * 1024ULL * 1024ULL * 1024ULL;
         if (gram_bytes >= hard_4d_gram_threshold) {
           while (!tryAllocateTiledPass1AndGram(tiled_tile_rows[n], rank_products[n])) {
             uint64_t old_rows = tiled_tile_rows[n];
             uint64_t new_rows = std::max<uint64_t>(1, old_rows / 2);
             if (new_rows == old_rows && old_rows > 1) --new_rows;
             if (new_rows == 0 || new_rows == old_rows) {
               restore_mode();
               return false;
             }
             tiled_tile_rows[n] = new_rows;
             rebuild_tiled_mode(n);
             std::cout << "  mode " << n
                       << ": late tiled pass-1 workspace + Gram probe failed, retrying with "
                       << new_rows << " root slices per tile -> "
                       << tiled_mode_tiles[n].size() << " tiles\n";
           }
         }
       }

       if (ttmc_caches[n].initialized) free_ttmc_cache(ttmc_caches[n]);
       return true;
     };

     auto trySVDStyleRescueForMode = [&](int n,
                                         const CSFCopy& csf,
                                         const std::vector<uint64_t>& ranks_v,
                                         const std::vector<scalar_t*>& d_factor_mats_v,
                                         uint64_t arr_O_size,
                                         uint64_t M,
                                         uint64_t N,
                                         double& ttmc_us_accum,
                                         double& svd_us_accum,
                                         bool full_y_already_materialized,
                                         double* mode0_core_norm_sq_out,
                                         bool* mode0_core_ready_out) -> bool {
       scalar_t* d_rescue_A = nullptr;
       bool allocated_full_y = false;
       double local_ttmc_us = 0.0;
       auto rescue_start = std::chrono::high_resolution_clock::now();

       if (full_y_already_materialized && d_arr_O != nullptr) {
         d_rescue_A = d_arr_O;
       } else {
         prepare_full_mode_cache(n, ranks_v);
         cudaError_t alloc_err = cudaMalloc(&d_rescue_A, sizeof(scalar_t) * arr_O_size);
         if (alloc_err != cudaSuccess) {
           cudaGetLastError();
           if (verbose) {
             std::cout << "  SVD-style rescue full-Y materialization alloc failed for mode "
                       << n << "\n";
           }
           return false;
         }
         allocated_full_y = true;
         auto ttmc_start = std::chrono::high_resolution_clock::now();
         run_ttmc_cuda(
           const_cast<uint64_t**>(csf.d_ptrs.data()),
           const_cast<uint64_t**>(csf.d_idxs.data()),
           csf.d_values,
           const_cast<scalar_t**>(d_factor_mats_v.data()),
           d_rescue_A, arr_O_size, /*ncm=*/0,
           const_cast<uint64_t*>(ranks_v.data()), order,
           ttmc_caches[n],
           /*log_method=*/verbose);
         auto ttmc_end = std::chrono::high_resolution_clock::now();
         local_ttmc_us = std::chrono::duration_cast<std::chrono::microseconds>(
           ttmc_end - ttmc_start).count();
       }

       try {
         runSVdStyleRescueFromFullMatrix(
           cusolverH, cublasH, d_rescue_A,
           static_cast<int>(M), static_cast<int>(N),
           static_cast<int>(ranks[n]), d_factors[n], verbose);

         if (n == 0 && mode0_core_norm_sq_out && mode0_core_ready_out) {
           scalar_t* d_core_tmp = nullptr;
           scalar_t core_norm = (scalar_t)0;
           scalar_t gemm_alpha = (scalar_t)1;
           scalar_t gemm_beta = (scalar_t)0;
           const uint64_t core_elems = (uint64_t)ranks[n] * N;
           CHECK_CUDA(cudaMalloc(&d_core_tmp, sizeof(scalar_t) * core_elems));
           CHECK_CUBLAS(cublasGemmT(
             cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
             (int)N, (int)ranks[n], (int)M,
             &gemm_alpha,
             d_rescue_A, (int)N,
             d_factors[n], (int)ranks[n],
             &gemm_beta, d_core_tmp, (int)N));
           CHECK_CUBLAS(cublasNrm2T(cublasH, (int)core_elems, d_core_tmp, 1, &core_norm));
           CHECK_CUDA(cudaFree(d_core_tmp));
           *mode0_core_norm_sq_out = (double)core_norm * (double)core_norm;
           *mode0_core_ready_out = true;
         }
       } catch (...) {
         if (allocated_full_y && d_rescue_A) {
           cudaFree(d_rescue_A);
           d_rescue_A = nullptr;
         }
         throw;
       }

       auto rescue_end = std::chrono::high_resolution_clock::now();
       double total_us = std::chrono::duration_cast<std::chrono::microseconds>(
         rescue_end - rescue_start).count();
       ttmc_us_accum += local_ttmc_us;
       svd_us_accum += (total_us - local_ttmc_us);

       if (allocated_full_y) {
         CHECK_CUDA(cudaFree(d_rescue_A));
       }
       return true;
     };

     for (iter = 0; iter < max_iters; iter++) {
       auto iter_start = std::chrono::high_resolution_clock::now();
       double mode0_core_norm_sq = 0.0;
       bool mode0_core_ready = false;

       for (int n = order - 1; n >= 0; n--) {
         uint64_t arr_O_size = arr_O_sizes[n];
         CSFCopy& csf = csf_copies[n];

         std::vector<uint64_t>  ranks_v(order);
         for (int l = 0; l < order; l++)
           ranks_v[l] = (l == 0) ? ranks[n] : ranks[csf.modeOrder[l]];

         std::vector<scalar_t*> d_factor_mats_v(order, nullptr);
         for (int l = 1; l < order; l++)
           d_factor_mats_v[l] = d_factors[csf.modeOrder[l]];

         if (!use_tiled[n]) {
           ensure_full_arr_O();
           prepare_full_mode_cache(n, ranks_v);
           auto ttmc_start = std::chrono::high_resolution_clock::now();
           run_ttmc_cuda(
             csf.d_ptrs.data(), csf.d_idxs.data(), csf.d_values,
             d_factor_mats_v.data(),
             d_arr_O, arr_O_size, /*ncm=*/0,
             ranks_v.data(), order,
             ttmc_caches[n],
             /*log_method=*/verbose);
           auto ttmc_end = std::chrono::high_resolution_clock::now();
           double ttmc_us = std::chrono::duration_cast<std::chrono::microseconds>(
             ttmc_end - ttmc_start).count();
           ttmc_time_us[n] += ttmc_us;
           if (verbose)
             std::cout << "[iter " << iter << " mode " << n << "] TTMc: " << ttmc_us << " us\n";

           if (check && iter == 0 && order == 3) {
             CHECK_CUDA(cudaMemcpy(arr_O_host, d_arr_O,
               sizeof(scalar_t) * arr_O_size, cudaMemcpyDeviceToHost));
             int idx_A = csf.modeOrder[1];
             int idx_B = csf.modeOrder[2];
             uint32_t f1 = static_cast<uint32_t>(ranks[idx_A]);
             uint32_t f2 = static_cast<uint32_t>(ranks[idx_B]);
             verify_ttmc(csf, factors[idx_A], factors[idx_B],
                         arr_O_host, coo.dims[n], f1, f2, n);
           }

           uint64_t M = coo.dims[n];
           uint64_t N = rank_products[n];
           auto svd_start = std::chrono::high_resolution_clock::now();
#if !SCALAR_DOUBLE
           const bool allow_late_tiled_fallback =
             ttmc_path == TTMcStorageMode::Auto &&
             supports_tiled[n] &&
             !use_tiled[n] &&
             !late_runtime_tiled_disabled[n];
           try {
             gpu_truncated_svd_update_factor(
               cusolverH, cublasH, d_arr_O,
               static_cast<int>(M), static_cast<int>(N),
               static_cast<int>(ranks[n]), d_factors[n], verbose,
               /*allow_iterative_after_exact=*/false);
           } catch (const ExactTopRFullGramFallbackFailed&) {
             if (!allow_late_tiled_fallback) throw;

             bool rescued = false;
             if (allow_late_tiled_fallback) {
               std::cout << "  full-Gram eigensolver failed for mode " << n
                         << ", trying late tiled fallback\n";
               if (tryPromoteModeToTiledAtRuntime(n, ranks_v)) {
                 auto failed_svd_end = std::chrono::high_resolution_clock::now();
                 double failed_svd_us = std::chrono::duration_cast<std::chrono::microseconds>(
                   failed_svd_end - svd_start).count();
                 svd_time_us[n] += failed_svd_us;

                 bool tiled_4d_mode = (order == 4);
                 if (tiled_4d_mode) {
                   release_full_mode_state_for_tiled_4d(n);
                 }
                TiledModeUpdateStats tiled_stats;
                try {
                  tiled_stats = runTiledModeUpdate(
                    cusolverH, cublasH,
                    tiled_mode_tiles[n],
                    tiled_workspace,
                    d_factor_mats_v.data(),
                    d_factors[n],
                    factor_sizes[n],
                    ranks_v.data(),
                    order,
                    (int)coo.dims[n],
                    tiled_4d_mode,
                    verbose);
                } catch (const TiledGramResolutionFailed&) {
                  if (verbose) {
                    std::cout << "  tiled Gram resolution failed for mode " << n
                              << ", trying SVD-style rescue\n";
                  }
                  bool rescued = trySVDStyleRescueForMode(
                    n, csf, ranks_v, d_factor_mats_v,
                    arr_O_size, M, N,
                    ttmc_time_us[n], svd_time_us[n],
                    /*full_y_already_materialized=*/(d_arr_O != nullptr),
                    &mode0_core_norm_sq, &mode0_core_ready);
                  if (!rescued) {
                    throw;
                  }
                  if (check && iter == 0) {
                    CHECK_CUDA(cudaMemcpy(factors[n], d_factors[n],
                      sizeof(scalar_t) * factor_sizes[n], cudaMemcpyDeviceToHost));
                  }
                  continue;
                }
                ttmc_time_us[n] += tiled_stats.ttmc_us;
                svd_time_us[n] += (tiled_stats.total_us - tiled_stats.ttmc_us);
                if (n == 0) {
                   mode0_core_norm_sq = tiled_stats.core_norm_sq;
                   mode0_core_ready = true;
                 }
                 if (check && iter == 0) {
                   CHECK_CUDA(cudaMemcpy(factors[n], d_factors[n],
                     sizeof(scalar_t) * factor_sizes[n], cudaMemcpyDeviceToHost));
                 }
                 if (verbose) {
                   std::cout << "[iter " << iter << " mode " << n
                             << "] late tiled fallback total: "
                             << tiled_stats.total_us << " us"
                             << "  TTMc-only: " << tiled_stats.ttmc_us << " us\n";
                 }
                 continue;
               }
               late_runtime_tiled_disabled[n] = true;
             }

             if (!rescued) {
               throw;
             }
           }
#else
           gpu_truncated_svd_update_factor(
             cusolverH, cublasH, d_arr_O,
             static_cast<int>(M), static_cast<int>(N),
             static_cast<int>(ranks[n]), d_factors[n], verbose);
#endif
           if (check && iter == 0) {
             CHECK_CUDA(cudaMemcpy(factors[n], d_factors[n],
               sizeof(scalar_t) * factor_sizes[n], cudaMemcpyDeviceToHost));
           }
           auto svd_end = std::chrono::high_resolution_clock::now();
           double svd_us = std::chrono::duration_cast<std::chrono::microseconds>(
             svd_end - svd_start).count();
           svd_time_us[n] += svd_us;
           if (verbose)
             std::cout << "  SVD: " << svd_us << " us\n";
         } else {
           bool tiled_4d_mode = (order == 4);
           if (tiled_4d_mode) {
             release_full_mode_state_for_tiled_4d(n);
           }
           if (check && iter == 0 && verbose) {
             std::cout << "[iter 0 mode " << n
                       << "] skipping CPU TTMc check in tiled mode because Y is never fully materialized\n";
           }
           TiledModeUpdateStats tiled_stats;
           try {
             tiled_stats = runTiledModeUpdate(
               cusolverH, cublasH,
               tiled_mode_tiles[n],
               tiled_workspace,
               d_factor_mats_v.data(),
               d_factors[n],
               factor_sizes[n],
               ranks_v.data(),
               order,
               (int)coo.dims[n],
               tiled_4d_mode,
               verbose);
           } catch (const TiledGramResolutionFailed&) {
             if (verbose) {
               std::cout << "  tiled Gram resolution failed for mode " << n
                         << ", trying SVD-style rescue\n";
             }
             bool rescued = trySVDStyleRescueForMode(
               n, csf, ranks_v, d_factor_mats_v,
               arr_O_size, coo.dims[n], rank_products[n],
               ttmc_time_us[n], svd_time_us[n],
               /*full_y_already_materialized=*/false,
               &mode0_core_norm_sq, &mode0_core_ready);
             if (!rescued) {
               throw;
             }
             if (check && iter == 0) {
               CHECK_CUDA(cudaMemcpy(factors[n], d_factors[n],
                 sizeof(scalar_t) * factor_sizes[n], cudaMemcpyDeviceToHost));
             }
             continue;
           }
           ttmc_time_us[n] += tiled_stats.ttmc_us;
           svd_time_us[n] += (tiled_stats.total_us - tiled_stats.ttmc_us);
           if (n == 0) {
             mode0_core_norm_sq = tiled_stats.core_norm_sq;
             mode0_core_ready = true;
           }
           if (check && iter == 0) {
             CHECK_CUDA(cudaMemcpy(factors[n], d_factors[n],
               sizeof(scalar_t) * factor_sizes[n], cudaMemcpyDeviceToHost));
           }
           if (verbose) {
             std::cout << "[iter " << iter << " mode " << n << "] Tiled total: "
                       << tiled_stats.total_us << " us"
                       << "  TTMc-only: " << tiled_stats.ttmc_us << " us\n";
           }
         }
       }

       if (verbose) {
         std::cout << "\n--- iter " << iter << " ---\n";
         for (int n = 0; n < order; n++)
           std::cout << "  Mode-" << n << ": TTMc " << ttmc_time_us[n]
                     << " us  SVD " << svd_time_us[n] << " us\n";
       }
       for (int n = 0; n < order; n++) { ttmc_time_us[n] = 0; svd_time_us[n] = 0; }

       scalar_t core_norm = (scalar_t)0;
       if (!use_tiled[0]) {
         ensure_full_arr_O();
         ensure_mode0_core_buffer();
         scalar_t gemm_alpha = (scalar_t)1, gemm_beta = (scalar_t)0;
         CHECK_CUBLAS(cublasGemmT(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
           N_rest, R0, I0, &gemm_alpha,
           d_arr_O,       N_rest,
           d_factors[0],  R0,
           &gemm_beta, d_G_core, N_rest));
         CHECK_CUBLAS(cublasNrm2T(cublasH, (int)(R0 * N_rest), d_G_core, 1, &core_norm));
       } else {
         if (!mode0_core_ready)
           throw std::runtime_error("Missing mode-0 singular values for tiled fit computation.");
         core_norm = std::sqrt((scalar_t)std::max(mode0_core_norm_sq, 0.0));
       }

       scalar_t norm_residual = std::sqrt(
         std::max((scalar_t)0, input_tsr_norm * input_tsr_norm - core_norm * core_norm));
       scalar_t fit       = (scalar_t)1 - norm_residual / input_tsr_norm;
       scalar_t delta_fit = std::fabs(fit - prev_fit);

       std::cout << "[iter " << iter << "]"
                 << "  core_norm=" << core_norm
                 << "  fit=" << fit
                 << "  delta_fit=" << delta_fit << "\n";

       auto iter_end = std::chrono::high_resolution_clock::now();
       auto iter_us = std::chrono::duration_cast<std::chrono::microseconds>(
         iter_end - iter_start).count();
       std::cout << "[HOOI Iter Time] iter=" << iter
                 << " runtime_us=" << iter_us << "\n";

      const bool hit_max_iters = (iter >= max_iters - 1);
      const bool hit_tol = (iter != 0 && delta_fit < tol);
      if (hit_max_iters || hit_tol) {
        if (hit_tol) {
          std::cout << "Converged (delta_fit " << delta_fit << " < tol " << tol << ")\n";
        } else {
          std::cout << "Reached max_iters (" << max_iters
                    << ") with delta_fit=" << delta_fit
                    << " and tol=" << tol << "\n";
        }
        iter++;
        break;
      }
      prev_fit = fit;
    }

     auto total_end = std::chrono::high_resolution_clock::now();
     auto total_us  = std::chrono::duration_cast<std::chrono::microseconds>(
       total_end - total_start).count();
     int num_iters = std::min(iter, max_iters);

     if (!use_tiled[0]) {
       ensure_full_arr_O();
       ensure_mode0_core_buffer();
       scalar_t a = (scalar_t)1, b = (scalar_t)0;
       CHECK_CUBLAS(cublasGemmT(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
         N_rest, R0, I0, &a,
         d_arr_O,       N_rest,
         d_factors[0],  R0,
         &b, d_G_core, N_rest));
     } else if (verbose) {
       std::cout << "Final core remains implicit in tiled mode-0 path as Sigma * V^T.\n";
     }

     for (int n = 0; n < order; n++) {
       if (!use_tiled[n]) free_ttmc_cache(ttmc_caches[n]);
       else freeTTMcTiles3D(tiled_mode_tiles[n]);
     }
     freeTiledModeWorkspace(tiled_workspace);
     if (d_G_core) CHECK_CUDA(cudaFree(d_G_core));
     if (d_arr_O) CHECK_CUDA(cudaFree(d_arr_O));
     for (int i = 0; i < order; i++) CHECK_CUDA(cudaFree(d_factors[i]));
     for (int n = 0; n < order; n++) freeCSFGPU(csf_copies[n]);
     if (arr_O_host) std::free(arr_O_host);
     for (int i = 0; i < order; i++) delete[] factors[i];
     cusolverDnDestroy(cusolverH);
     cublasDestroy(cublasH);
 
     std::cout << "\nTucker HOOI done: " << num_iters << " iters, " << total_us << " us total\n";
     return 0;
 
   } catch (const std::exception& e) {
     std::cerr << "Error: " << e.what() << "\n";
     return 1;
   }
 }
 
