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
  scalar_t* d_A, int M, int N, int R, scalar_t* d_factor, bool verbose) {
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
     scalar_t* d_Gram;
     cudaEventRecord(ev_start);
     CHECK_CUDA(cudaMalloc(&d_Gram, sizeof(scalar_t) * N * N));
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  ATA alloc: " << ev_ms << " ms\n";

     cudaEventRecord(ev_start);
     CHECK_CUBLAS(cublasGemmT(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
       N, N, M, &alpha, d_A, N, d_A, N, &beta, d_Gram, N));
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  ATA gemm: " << ev_ms << " ms\n";

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
       gpu_full_svd_update_factor(cusolverH, cublasH, d_A, M, N, R, d_factor, verbose);
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
       cublasStatus_t gemm_st = cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
         N, N, M, &alpha_dp, d_A_dp, N, d_A_dp, N, &beta_zero, d_Gram_dp, N);
       CHECK_CUDA(cudaFree(d_A_dp));
       if (gemm_st != CUBLAS_STATUS_SUCCESS) {
         if (verbose) std::cout << "  full dp gemm failed (status=" << gemm_st
                                << "), falling back to blocked\n";
         use_blocked = true;
       }
     } else {
       cudaGetLastError();
       if (verbose) std::cout << "  full dp alloc failed, using blocked path\n";
       use_blocked = true;
     }
     if (use_blocked) {
       const int BSIZ = 32768;
       double* d_blk_dp;
       CHECK_CUDA(cudaMalloc(&d_blk_dp, sizeof(double) * std::min(BSIZ, M) * N));
       double beta_one = 1.0;
       for (int s = 0; s < M; s += BSIZ) {
         int b = std::min(BSIZ, M - s);
         long long chunk_elems = (long long)b * N;
         convert_scalar_to_dp<<<(chunk_elems + 255)/256, 256>>>(
           d_A + (long long)s * N, d_blk_dp, chunk_elems);
         CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
           N, N, b, &alpha_dp, d_blk_dp, N, d_blk_dp, N,
           (s == 0) ? &beta_zero : &beta_one, d_Gram_dp, N));
       }
       CHECK_CUDA(cudaFree(d_blk_dp));
     }
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  ATA gemm (dp): " << ev_ms << " ms\n";

     // trace(A^T A) = ||A||_F^2  — dasum on diagonal (all entries >= 0 for Gram)
     double traceATA;
     CHECK_CUBLAS(cublasDasum(cublasH, N, d_Gram_dp, N + 1, &traceATA));
     if (verbose) std::cout << "  trace(ATA): " << traceATA << "\n";

     double *d_W_dp;
     CHECK_CUDA(cudaMalloc(&d_W_dp, sizeof(double) * N));
     int lwork = 0;
     CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
       CUBLAS_FILL_MODE_UPPER, N, d_Gram_dp, N, d_W_dp, &lwork));

     double *d_work_dp; int *d_info;
     cudaEventRecord(ev_start);
     CHECK_CUDA(cudaMalloc(&d_work_dp, sizeof(double) * lwork));
     CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  eig buf alloc: " << ev_ms << " ms\n";
     if (verbose) std::cout << "  d_work (MB): " << sizeof(double) * lwork / (1024.0 * 1024.0) << "\n";

     cudaEventRecord(ev_start);
     CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
       CUBLAS_FILL_MODE_UPPER, N, d_Gram_dp, N, d_W_dp, d_work_dp, lwork, d_info));
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  eig decomp (dp): " << ev_ms << " ms\n";
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
     scalar_t* d_V_R;
     CHECK_CUDA(cudaMalloc(&d_V_R, sizeof(scalar_t) * N * R));
     convert_dp_to_scalar<<<((long long)N*R + 255)/256, 256>>>(d_V_R_dp, d_V_R, N * R);

     cudaEventRecord(ev_start);
     CHECK_CUBLAS(cublasGemmT(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
       R, M, N, &alpha, d_V_R, N, d_A, N, &beta, d_factor, R));
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  AV gemm: " << ev_ms << " ms\n";

     scalar_t* d_diag;
     CHECK_CUDA(cudaMalloc(&d_diag, sizeof(scalar_t) * R));
     cudaEventRecord(ev_start);
     compute_inv_sigma_dp<<<(R+255)/256, 256>>>(d_W_dp, d_diag, N, R, 1e-12);
     CHECK_CUBLAS(cublasDgmmT(cublasH, CUBLAS_SIDE_LEFT, R, M,
       d_factor, R, d_diag, 1, d_factor, R));
     cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
     cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
     if (verbose) std::cout << "  normalize (1/sigma): " << ev_ms << " ms\n";

     CHECK_CUDA(cudaFree(d_diag));
     CHECK_CUDA(cudaFree(d_V_R));
     CHECK_CUDA(cudaFree(d_Gram_dp)); CHECK_CUDA(cudaFree(d_W_dp));
     CHECK_CUDA(cudaFree(d_work_dp)); CHECK_CUDA(cudaFree(d_info));
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
  uint64_t output_elems = 0;

  std::vector<uint64_t> h_local_rows;
  std::vector<uint64_t> h_mode1_ptr;
  std::vector<uint64_t> h_mode2_ptr;

  uint64_t* d_local_rows = nullptr;
  uint64_t* d_mode1_ptr = nullptr;
  uint64_t* d_mode2_ptr = nullptr;

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
  uint64_t max_rows = 0;
  int cols = 0;
  int rank = 0;
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
#endif
  ws.max_rows = 0;
  ws.cols = 0;
  ws.rank = 0;
}

static void prepareTiledModeWorkspace(
  TiledModeWorkspace& ws,
  uint64_t max_rows,
  int cols,
  int rank)
{
  freeTiledModeWorkspace(ws);
  ws.max_rows = max_rows;
  ws.cols = cols;
  ws.rank = rank;
  if (max_rows == 0) return;
  CHECK_CUDA(cudaMalloc(&ws.d_y_tile, sizeof(scalar_t) * max_rows * (uint64_t)cols));
  CHECK_CUDA(cudaMalloc(&ws.d_factor_tile, sizeof(scalar_t) * max_rows * (uint64_t)rank));
#if !SCALAR_DOUBLE
  ws.dp_block_rows = (int)std::min<uint64_t>(max_rows, 8192);
  CHECK_CUDA(cudaMalloc(
    &ws.d_tile_block_dp,
    sizeof(double) * (uint64_t)ws.dp_block_rows * (uint64_t)cols));
#endif
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
  tile.d_global_rows = nullptr;
  tile.d_values = nullptr;
  tile.d_ptrs.clear();
  tile.d_idxs.clear();
  tile.h_local_rows.clear();
  tile.h_mode1_ptr.clear();
  tile.h_mode2_ptr.clear();
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
  for (uint64_t start = 0; start < tile_rows; start += (uint64_t)ws.dp_block_rows) {
    int block_rows = (int)std::min<uint64_t>((uint64_t)ws.dp_block_rows, tile_rows - start);
    long long elems = (long long)block_rows * cols;
    convert_scalar_to_dp<<<(elems + 255) / 256, 256>>>(
      ws.d_y_tile + start * (uint64_t)cols,
      ws.d_tile_block_dp,
      elems);
    CHECK_CUDA(cudaGetLastError());
    double beta = gram_initialized ? 1.0 : 0.0;
    CHECK_CUBLAS(cublasDgemm(
      cublasH,
      CUBLAS_OP_N, CUBLAS_OP_T,
      cols, cols, block_rows,
      &alpha,
      ws.d_tile_block_dp, cols,
      ws.d_tile_block_dp, cols,
      &beta,
      d_gram_dp, cols));
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
  CHECK_CUBLAS(cublasGemmT(
    cublasH,
    CUBLAS_OP_N, CUBLAS_OP_T,
    cols, cols, (int)tile_rows,
    &alpha,
    ws.d_y_tile, cols,
    ws.d_y_tile, cols,
    &beta,
    d_gram, cols));
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
  bool verbose)
{
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
  CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
    cols, d_gram, cols, d_W, &lwork));
  CHECK_CUDA(cudaMalloc(&d_work, sizeof(double) * lwork));
  CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
  CHECK_CUSOLVER(cusolverDnDsyevd(
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
    check_resolution_dp<<<1, rank - 1>>>(d_W, M_total + cols, cols, rank, eps_mach_dp, d_info, traceATA);
  CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info != 0) {
    if (verbose)
      std::cout << "  tiled Gram resolution check failed; full fallback is unavailable in tiled mode\n";
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
#endif

  return true;
}

static TiledModeUpdateStats runTiledModeUpdate3D(
  cusolverDnHandle_t cusolverH,
  cublasHandle_t cublasH,
  std::vector<TTMCTile3D>& tiles,
  const TiledModeWorkspace& ws,
  scalar_t** d_factor_mats_v,
  scalar_t* d_factor_out,
  uint64_t factor_size_out,
  uint64_t* ranks_v,
  int M_total,
  bool verbose)
{
  TiledModeUpdateStats stats;
  const int cols = (int)(ranks_v[1] * ranks_v[2]);
  const int rank = (int)ranks_v[0];
  scalar_t alpha = (scalar_t)1;
  scalar_t beta = (scalar_t)0;

  auto total_start = std::chrono::high_resolution_clock::now();

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
      ws.d_y_tile, tile.output_elems,
      /*ncm=*/0, ranks_v, /*order=*/3,
      tile.cache, /*log_method=*/false);
    auto t1 = std::chrono::high_resolution_clock::now();
    stats.ttmc_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    accumulateGramFromTile(cublasH, ws, d_gram, tile.num_rows, cols, gram_initialized);
  }

  TiledGramSVDResult svd_result;
  if (!solveTallGramEigensystem(cusolverH, cublasH, d_gram, M_total, cols, rank, svd_result, verbose)) {
#if SCALAR_DOUBLE
    CHECK_CUDA(cudaFree(d_gram));
#else
    CHECK_CUDA(cudaFree(d_gram));
#endif
    throw std::runtime_error("Tiled Gram eigensolve failed; full fallback is unavailable in tiled mode.");
  }

  CHECK_CUDA(cudaFree(d_gram));
  CHECK_CUDA(cudaMemset(d_factor_out, 0, sizeof(scalar_t) * factor_size_out));

  for (auto& tile : tiles) {
    auto t0 = std::chrono::high_resolution_clock::now();
    run_ttmc_cuda(
      tile.d_ptrs.data(), tile.d_idxs.data(), tile.d_values,
      d_factor_mats_v,
      ws.d_y_tile, tile.output_elems,
      /*ncm=*/0, ranks_v, /*order=*/3,
      tile.cache, /*log_method=*/false);
    auto t1 = std::chrono::high_resolution_clock::now();
    stats.ttmc_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    CHECK_CUBLAS(cublasGemmT(
      cublasH,
      CUBLAS_OP_T, CUBLAS_OP_N,
      rank, (int)tile.num_rows, cols,
      &alpha,
      svd_result.d_VR, cols,
      ws.d_y_tile, cols,
      &beta,
      ws.d_factor_tile, rank));
    CHECK_CUBLAS(cublasDgmmT(
      cublasH,
      CUBLAS_SIDE_LEFT,
      rank, (int)tile.num_rows,
      ws.d_factor_tile, rank,
      svd_result.d_inv_sigma, 1,
      ws.d_factor_tile, rank));

    uint64_t total_scatter = tile.num_rows * (uint64_t)rank;
    scatter_factor_rows_kernel<<<(total_scatter + 255) / 256, 256>>>(
      tile.d_global_rows, ws.d_factor_tile, d_factor_out, tile.num_rows, rank);
    CHECK_CUDA(cudaGetLastError());
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  auto total_end = std::chrono::high_resolution_clock::now();
  stats.total_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
  stats.core_norm_sq = svd_result.core_norm_sq;
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
          << "  --tile-mb N           Approximate Y-tile buffer size in MiB (default 256)\n";
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
 
     std::cout << "Tensor:";
     for (int i = 0; i < order; i++) std::cout << " " << coo.dims[i];
     std::cout << "  nnz=" << coo.values.size() << "  ranks=(";
     for (int i = 0; i < order; i++) { if (i) std::cout << ","; std::cout << ranks[i]; }
     std::cout << ")\n";
 
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
     for (int i = 0; i < order; i++) {
       factor_sizes[i] = coo.dims[i] * ranks[i];
       factors[i] = new scalar_t[factor_sizes[i]];
       init_factor_orthonormal(coo.dims[i], ranks[i], 42 + i, factors[i]);
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

     if (ttmc_path == TTMcStorageMode::Tiled && order != 3) {
       throw std::runtime_error("The tiled/recompute path is currently implemented only for 3D tensors.");
     }

     std::vector<bool> use_tiled(order, false);
     std::vector<bool> supports_tiled(order, false);
     for (int n = 0; n < order; n++) {
       supports_tiled[n] = (order == 3) && (coo.dims[n] > rank_products[n]);
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

     if (ttmc_path == TTMcStorageMode::Auto) {
       size_t free_bytes = 0, total_bytes = 0;
       CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
       const size_t reserve_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
       uint64_t max_full_arr_O_size = recompute_max_full_arr_O_size();
       while (max_full_arr_O_size > 0 &&
              (size_t)max_full_arr_O_size * sizeof(scalar_t) + reserve_bytes > free_bytes) {
         int mode_to_tile = choose_mode_to_tile();
         if (mode_to_tile < 0) break;
         use_tiled[mode_to_tile] = true;
         max_full_arr_O_size = recompute_max_full_arr_O_size();
       }
     }

     uint64_t max_full_arr_O_size = recompute_max_full_arr_O_size();

     scalar_t* d_arr_O = nullptr;
     while (max_full_arr_O_size > 0) {
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
     std::vector<TiledModeWorkspace> tiled_workspaces(order);
     std::vector<uint64_t> tiled_tile_rows(order, 0);
     const size_t tile_reserve_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
     int tiled_modes_remaining = 0;
     for (int n = 0; n < order; n++)
       if (use_tiled[n]) ++tiled_modes_remaining;

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
         prepareTTMcTiles3D(
           csf,
           ranks_v.data(), fsizes_v.data(), dims_v.data(),
           tile_rows,
           tiled_mode_tiles[n]);
         prepareTiledModeWorkspace(
           tiled_workspaces[n],
           tile_rows,
           (int)rank_products[n],
           (int)ranks[n]);
         std::cout << "  mode " << n << ": tiled into " << tiled_mode_tiles[n].size()
                   << " tiles of up to " << tile_rows << " root slices"
                   << " (tile budget " << formatBytes((double)tile_target_bytes) << ")\n";
         --tiled_modes_remaining;
       }
     }

     // ===================================================================
     // 8. HOOI loop
     // ===================================================================
     scalar_t prev_fit = (scalar_t)0;
     int iter;
     std::vector<double> ttmc_time_us(order, 0.0);
     std::vector<double> svd_time_us(order, 0.0);
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

     for (iter = 0; iter < max_iters; iter++) {
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
           gpu_truncated_svd_update_factor(cusolverH, cublasH, d_arr_O,
             static_cast<int>(M), static_cast<int>(N),
             static_cast<int>(ranks[n]), d_factors[n], verbose);
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
           if (check && iter == 0 && verbose) {
             std::cout << "[iter 0 mode " << n
                       << "] skipping CPU TTMc check in tiled mode because Y is never fully materialized\n";
           }
           TiledModeUpdateStats tiled_stats = runTiledModeUpdate3D(
             cusolverH, cublasH,
             tiled_mode_tiles[n],
             tiled_workspaces[n],
             d_factor_mats_v.data(),
             d_factors[n],
             factor_sizes[n],
             ranks_v.data(),
             (int)coo.dims[n],
             verbose);
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

       if (
        iter>=max_iters - 1
        // !(iter == 0) && delta_fit < tol
      ) {
         std::cout << "Converged (delta_fit " << delta_fit << " < tol " << tol << ")\n";
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
       freeTiledModeWorkspace(tiled_workspaces[n]);
     }
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
 
