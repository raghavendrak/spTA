#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <cctype>
#include <strings.h>
#include <cuda_runtime.h>
#include "csf_tensor.h"
#include "matrix_utils.h"

using namespace std;

/*
High-level code flow (reader guide, with function names and CPU/GPU ownership):
1) CPU: main()
   - readCSFTensor() loads CSF input from disk.
   - getCSFArrays() exposes CSF pointer/index arrays.
   - generate_matrix() builds dense factor matrices.
   - run_ttmc_cuda() performs GPU orchestration and launch selection.
2) CPU: run_ttmc_cuda()
   - cudaMalloc/cudaMemcpy stages sparse metadata + factors on device.
   - For ncm==0, order==3:
     * analyzeFiberStats() computes per-j fiber statistics.
     * chooseDynamicHints() proposes base dynamic launch/tile settings.
     * Selector picks kernel family:
       ultra_tiny_k or long_tail_sparse -> kernel_ttmc3_static_block_per_i()
       medium_fiber_dynamic_tuned or default
         -> buildDynamicTaskRanges() + kernel_ttmc3_dynamic_tasks()
     * Dynamic branch may retune launch hints (block/grid/tile) for comm-like tensors.
   - For ncm==0, order==4:
     * analyzeFiberStats() + tiny-selector heuristics estimate cost.
     * Tiny path: launch_ttmc4_tiny_streams_flat() -> kernel_ttmc4_tiny_stream().
     * Default path: buildDynamicTaskRanges() + kernel_ttmc4_dynamic_tasks().
3) GPU: __global__ kernels
   - kernel_ttmc3_static_block_per_i(): one block per i-fiber, warp-local accumulation.
   - kernel_ttmc3_dynamic_tasks()/kernel_ttmc4_dynamic_tasks(): warp-dequeued task queue
     for load-balanced processing of irregular fibers.
   - kernel_ttmc4_tiny_stream(): flattened microkernel for tiny 4D inputs.
4) CPU: run_ttmc_cuda() copies output back and frees device allocations.
5) CPU: main() optionally calls cpu_factorize_n_fuse() for --verify.
*/

// Helper macro for checking CUDA errors
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

/////////////////////////////////////////////////////////////////////
/* Start of GPU kernel definitions */
__global__ void kernel_ttmc3_static_block_per_i( 
  const uint64_t* __restrict__ mode_0_idx,
  const uint64_t* __restrict__ mode_1_ptr, const uint64_t* __restrict__ mode_1_idx,
  const uint64_t* __restrict__ mode_2_ptr, const uint64_t* __restrict__ mode_2_idx,
  const float* __restrict__ values, float* arr_A,  float* arr_B,  float* arr_O,
  uint32_t f1, uint32_t f2,  int num_warps)
{
  extern __shared__ float buf[];
  __shared__ int s_counter;
  // int buf_size = num_warps * f2     +    f1 * f2 ;
  int buf_index;

  uint64_t i_ptr = blockIdx.x;
  uint64_t i =  mode_0_idx[i_ptr];

  uint32_t warp_size = 32;
  uint32_t warp_id = threadIdx.x / warp_size;
  int tid_in_warp = threadIdx.x % warp_size;

  // buf_index = threadIdx.x;
  // if(buf_index < f1 * f2){
  //   buf[num_warps * f2 + buf_index] = 0.0;
  // }
  for(int buf_idx = threadIdx.x; buf_idx < f1 * f2; buf_idx += blockDim.x){
    buf[num_warps * f2 + buf_idx] = 0.0;
  }
  if (threadIdx.x == 0) s_counter = 0;   // initialize once per block
  __syncthreads();

  // for(uint64_t j_ptr_offset = mode_1_ptr[i_ptr]; j_ptr_offset < mode_1_ptr[i_ptr + 1]; j_ptr_offset += num_warps){
  uint64_t offset, j_ptr,j_ptr_offset =  mode_1_ptr[i_ptr];
  unsigned int full_mask = 0xFFFFFFFFu;

  while(true){
    if(tid_in_warp == 0) offset = atomicAdd(&s_counter, 1);
    offset = __shfl_sync(full_mask, offset, 0); // broadcast the offset to all threads in the warp
    j_ptr = j_ptr_offset + offset;
    if(j_ptr < mode_1_ptr[i_ptr + 1]){
      uint64_t j = mode_1_idx[j_ptr];

      // for(int buf_index = threadIdx.x; buf_index < buf_size; buf_index+= blockDim.x)
      //   buf[buf_index] = 0.0;
      for(int buf_idx_offset = warp_id * f2; buf_idx_offset < (warp_id + 1)* f2; buf_idx_offset += warp_size){
        buf_index = buf_idx_offset + tid_in_warp;
        if(buf_index < (warp_id + 1)* f2){
          buf[buf_index] = 0.0;
        }
      }
      // __syncthreads();

      for(uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr){
        uint64_t k = mode_2_idx[k_ptr];

        for(uint32_t s_offset = 0; s_offset < f2; s_offset += warp_size){
          uint32_t s = s_offset + tid_in_warp;
          if(s < f2){
            // atomicAdd(&buf[warp_id * f2 + s], values[k_ptr] * arr_B[k * f2 + s]);
            buf[warp_id * f2 + s] += values[k_ptr] * arr_B[k * f2 + s];
          }
        }
      }
      // __syncthreads(); - not required because single warp execute the code serially

      for(uint32_t r = 0; r < f1; ++r){
        for(uint32_t s_offset = 0; s_offset < f2; s_offset += warp_size){
          uint32_t s = s_offset + tid_in_warp;
          if(s < f2){
            // atomicAdd(&arr_O[i * f1* f2 + r * f2 + s], buf[warp_id * f2 + s] * arr_A[j * f1 + r]);
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
          // atomicAdd(&arr_O[i * f1* f2 + r * f2 + s], buf[num_warps * f2 + r * f2 + s]);
        }
      }
    }
  }
}

__global__ void kernel_ttmc3_dynamic_tasks(
  const uint64_t* __restrict__ mode_0_idx,
  const uint64_t* __restrict__ mode_1_ptr, const uint64_t* __restrict__ mode_1_idx,
  const uint64_t* __restrict__ mode_2_ptr, const uint64_t* __restrict__ mode_2_idx,
  const float* __restrict__ values, const float* __restrict__ arr_A, const float* __restrict__ arr_B,
  float* __restrict__ arr_O, uint32_t f1, uint32_t f2,
  const TaskRange* __restrict__ tasks, uint64_t num_tasks,
  unsigned long long* __restrict__ global_task_counter)
{
  extern __shared__ float shared_buf[];
  const uint32_t warp_size = 32;
  const uint32_t warp_id = threadIdx.x / warp_size;
  const uint32_t lane = threadIdx.x % warp_size;
  const uint32_t warps_per_block = blockDim.x / warp_size;

  const bool use_register_accum = (f1 <= 64) && (f2 <= warp_size);
  float* warp_temp = nullptr;
  float* warp_accum = nullptr;
  if (!use_register_accum) {
    warp_temp = shared_buf + warp_id * f2;
    warp_accum = shared_buf + warps_per_block * f2 + warp_id * (size_t)f1 * f2;
  }

  const unsigned full_mask = 0xFFFFFFFFu;
  const uint32_t rs = f1 * f2;

  if (use_register_accum) {
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
      float* out_base = arr_O + (i * (uint64_t)f1 * f2);

      float accum_local[64];
      if (lane < f2) {
#pragma unroll
        for (uint32_t r = 0; r < f1; ++r) {
          accum_local[r] = 0.0f;
        }
      }

      if (k_tile_begin < k_tile_end) {
        uint64_t j_ptr = j_begin;
        uint64_t j = mode_1_idx[j_ptr];
        const float* __restrict__ a_row = arr_A + (j * (uint64_t)f1);
        float temp_reg = 0.0f;
        if (lane < f2) {
          temp_reg = 0.0f;
        }

        for (uint64_t k_ptr = k_tile_begin; k_ptr < k_tile_end; ++k_ptr) {
          uint64_t k = mode_2_idx[k_ptr];
          float val = values[k_ptr];
          const float* __restrict__ b_row = arr_B + (k * (uint64_t)f2);
          if (lane < f2) {
            float b_val = b_row[lane];
            temp_reg = fmaf(val, b_val, temp_reg);
          }
        }

#pragma unroll
        for (uint32_t r = 0; r < f1; ++r) {
          float a = 0.0f;
          if (lane == 0) {
            a = a_row[r];
          }
          a = __shfl_sync(full_mask, a, 0);
          if (lane < f2) {
            accum_local[r] = fmaf(temp_reg, a, accum_local[r]);
          }
        }
      }
      else {
        for (uint64_t j_ptr = j_begin; j_ptr < j_end; ++j_ptr) {
          uint64_t j = mode_1_idx[j_ptr];
          const float* __restrict__ a_row = arr_A + (j * (uint64_t)f1);
          float temp_reg = 0.0f;
          if (lane < f2) {
            temp_reg = 0.0f;
          }

          for (uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
            uint64_t k = mode_2_idx[k_ptr];
            float val = values[k_ptr];
            const float* __restrict__ b_row = arr_B + (k * (uint64_t)f2);
            if (lane < f2) {
              float b_val = b_row[lane];
              temp_reg = fmaf(val, b_val, temp_reg);
            }
          }

#pragma unroll
          for (uint32_t r = 0; r < f1; ++r) {
            float a = 0.0f;
            if (lane == 0) {
              a = a_row[r];
            }
            a = __shfl_sync(full_mask, a, 0);
            if (lane < f2) {
              accum_local[r] = fmaf(temp_reg, a, accum_local[r]);
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
      float* out_base = arr_O + (i * (uint64_t)f1 * f2);

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
          float val = values[k_ptr];
          const float* __restrict__ b_row = arr_B + (k * (uint64_t)f2);
          for (uint32_t s = lane; s < f2; s += warp_size) {
            float b_val = b_row[s];
            warp_temp[s] = fmaf(val, b_val, warp_temp[s]);
          }
        }
        __syncwarp(full_mask);

#pragma unroll
        for (uint32_t r = 0; r < f1; ++r) {
          float a = 0.0f;
          if (lane == 0) {
            const float* __restrict__ a_row = arr_A + (j * (uint64_t)f1);
            a = a_row[r];
          }
          a = __shfl_sync(full_mask, a, 0);
          uint32_t base = r * f2;
          for (uint32_t s = lane; s < f2; s += warp_size) {
            warp_accum[base + s] = fmaf(warp_temp[s], a, warp_accum[base + s]);
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
            float val = values[k_ptr];
            const float* __restrict__ b_row = arr_B + (k * (uint64_t)f2);
            for (uint32_t s = lane; s < f2; s += warp_size) {
              float b_val = b_row[s];
              warp_temp[s] = fmaf(val, b_val, warp_temp[s]);
            }
          }
          __syncwarp(full_mask);

#pragma unroll
          for (uint32_t r = 0; r < f1; ++r) {
            float a = 0.0f;
            if (lane == 0) {
              const float* __restrict__ a_row = arr_A + (j * (uint64_t)f1);
              a = a_row[r];
            }
            a = __shfl_sync(full_mask, a, 0);
            uint32_t base = r * f2;
            for (uint32_t s = lane; s < f2; s += warp_size) {
              warp_accum[base + s] = fmaf(warp_temp[s], a, warp_accum[base + s]);
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
  const float* __restrict__ values,
  const float* __restrict__ arr_A,
  const float* __restrict__ arr_B,
  const float* __restrict__ arr_C,
  float* __restrict__ out_base,
  float* __restrict__ warp_temp,
  float* __restrict__ warp_b,
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
      const float* __restrict__ b_row = arr_B + (k * (uint64_t)f2);
      for (uint32_t r2 = lane; r2 < f2; r2 += warp_size) {
        warp_b[r2] = b_row[r2];
      }
      __syncwarp(mask);

      uint64_t l_begin = mode_3_ptr[k_ptr];
      uint64_t l_end = mode_3_ptr[k_ptr + 1];
      for (uint64_t l_ptr = l_begin; l_ptr < l_end; ++l_ptr) {
        uint64_t l = mode_3_idx[l_ptr];
        float val = values[l_ptr];
        const float* __restrict__ c_row = arr_C + (l * (uint64_t)f3);
        for (uint32_t idx = lane; idx < rs; idx += warp_size) {
          uint32_t r2 = idx / f3;
          uint32_t r3 = idx % f3;
          float scaled_b = warp_b[r2] * val;
          warp_temp[idx] = fmaf(scaled_b, c_row[r3], warp_temp[idx]);
        }
        __syncwarp(mask);
      }
    }
  }

  uint64_t j = mode_1_idx[j_ptr];
  const float* __restrict__ a_row = arr_A + (j * (uint64_t)f1);
  for (uint32_t r1 = 0; r1 < f1; ++r1) {
    float a = (lane == 0) ? a_row[r1] : 0.0f;
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
  const float* __restrict__ values,
  const float* __restrict__ arr_A, const float* __restrict__ arr_B, const float* __restrict__ arr_C,
  float* __restrict__ arr_O,
  uint32_t f1, uint32_t f2, uint32_t f3,
  const TaskRange* __restrict__ tasks, uint64_t num_tasks,
  unsigned long long* __restrict__ global_task_counter)
{
  extern __shared__ float shared_buf[];
  const uint32_t warp_size = 32;
  const uint32_t warp_id = threadIdx.x / warp_size;
  const uint32_t lane = threadIdx.x % warp_size;
  const uint32_t warps_per_block = blockDim.x / warp_size;
  const unsigned mask = 0xFFFFFFFFu;
  const size_t rs = (size_t)f2 * f3;

  float* warp_temp = shared_buf + warp_id * rs;
  float* warp_b = shared_buf + (size_t)warps_per_block * rs + warp_id * (size_t)f2;

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
    float* out_base = arr_O + (i * (uint64_t)f1 * (uint64_t)rs);

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

__constant__ uint64_t tiny4d_ofst[8];
__global__ void kernel_ttmc4_tiny_stream(
  uint64_t* meta_data, const float* __restrict__ values,
  const float* __restrict__ factor_matrices, const uint64_t* __restrict__ fact_ofst,
  float* arr_O, const uint64_t* __restrict__ ranks, int ncm, int order, 
  uint64_t j_ptr_offset, uint64_t i)
{
  extern __shared__ float buf[];
  uint64_t j_ptr = j_ptr_offset + blockIdx.x;
  uint64_t j = meta_data[tiny4d_ofst[3] + j_ptr];

  int buf_ofst = ranks[2] * ranks[3];

  for (int buf_index = threadIdx.y * blockDim.x + threadIdx.x; 
       buf_index < buf_ofst; buf_index += blockDim.x * blockDim.y) {
    buf[buf_index] = 0.0f;
  }
  __syncthreads();

  for (uint64_t k_ptr = meta_data[tiny4d_ofst[4] + j_ptr];
       k_ptr < meta_data[tiny4d_ofst[4] + j_ptr + 1]; ++k_ptr) {
    uint64_t k = meta_data[tiny4d_ofst[5] + k_ptr];

    int buf_index = threadIdx.y * blockDim.x + threadIdx.x;
    if (buf_index < ranks[3]) {
      buf[buf_ofst + buf_index] = 0.0f;
    }
    __syncthreads();

    for (uint64_t l_ptr_ofst = meta_data[tiny4d_ofst[6] + k_ptr];
         l_ptr_ofst < meta_data[tiny4d_ofst[6] + k_ptr + 1];
         l_ptr_ofst += blockDim.y) {
      uint64_t l_ptr = l_ptr_ofst + threadIdx.y;
      if (l_ptr < meta_data[tiny4d_ofst[6] + k_ptr + 1]) {
        uint64_t l = meta_data[tiny4d_ofst[7] + l_ptr];
        for (uint32_t t_ofst = 0; t_ofst < ranks[3]; t_ofst += blockDim.x) {
          uint32_t t = t_ofst + threadIdx.x;
          if (t < ranks[3]) {
            atomicAdd(&buf[buf_ofst + t], values[l_ptr] *
              factor_matrices[fact_ofst[2] + l * ranks[3] + t]);
          }
        }
      }
    }
    __syncthreads();

    for (uint32_t s_ofst = 0; s_ofst < ranks[2]; s_ofst += blockDim.y) {
      uint32_t s = s_ofst + threadIdx.y;
      if (s < ranks[2]) {
        for (uint32_t t_ofst = 0; t_ofst < ranks[3]; t_ofst += blockDim.x) {
          uint32_t t = t_ofst + threadIdx.x;
          if (t < ranks[3]) {
            atomicAdd(&buf[s * ranks[3] + t], buf[buf_ofst + t] *
              factor_matrices[fact_ofst[1] + k * ranks[2] + s]);
          }
        }
      }
    }
    __syncthreads();
  }
  __syncthreads();

  for (uint32_t r = 0; r < ranks[1]; ++r) {
    for (uint32_t s_ofst = 0; s_ofst < ranks[2]; s_ofst += blockDim.y) {
      uint32_t s = s_ofst + threadIdx.y;
      if (s < ranks[2]) {
        for (uint32_t t_ofst = 0; t_ofst < ranks[3]; t_ofst += blockDim.x) {
          uint32_t t = t_ofst + threadIdx.x;
          if (t < ranks[3]) {
            atomicAdd(&arr_O[ i * ranks[1] * ranks[2] * ranks[3]
              + r * ranks[2] * ranks[3]
              + s * ranks[3]
              + t],
              buf[s * ranks[3] + t] * factor_matrices[fact_ofst[0] + j * ranks[1] + r]);
          }
        }
      }
    }
  }
}

static double launch_ttmc4_tiny_streams_flat(
  uint64_t** mode_ptrs, uint64_t** mode_idxs,
  const float* values,
  float** factor_matrices, const uint64_t* factor_sizes,
  float* d_arr_O,
  int ncm, uint64_t* ranks, int order,
  const uint64_t size_mode_ptr[], const uint64_t size_mode_idx[],
  const uint64_t* dimensions,
  int stream_hint)
{
  uint64_t num_i = size_mode_ptr[1] - 1;
  if (num_i == 0) {
    return 0.0;
  }

  uint64_t offset[8] = {0};
  uint64_t meta_size = 0;
  for (int i = 0; i < order; ++i) {
    offset[2 * i] = meta_size;
    meta_size += size_mode_ptr[i];
    offset[2 * i + 1] = meta_size;
    meta_size += size_mode_idx[i];
  }
  cudaCheckError(cudaMemcpyToSymbol(tiny4d_ofst, offset, sizeof(uint64_t) * 2 * order));

  uint64_t* d_meta_data = nullptr;
  cudaCheckError(cudaMalloc(&d_meta_data, sizeof(uint64_t) * meta_size));
  for (int i = 0; i < order; ++i) {
    cudaCheckError(cudaMemcpy(d_meta_data + offset[2 * i], mode_ptrs[i],
                              sizeof(uint64_t) * size_mode_ptr[i], cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_meta_data + offset[2 * i + 1], mode_idxs[i],
                              sizeof(uint64_t) * size_mode_idx[i], cudaMemcpyHostToDevice));
  }

  uint64_t fact_offsets[3] = {0};
  uint64_t fact_size = 0;
  int idx = 0;
  for (int i = 0; i < order; ++i) {
    if (i == ncm) continue;
    fact_offsets[idx++] = fact_size;
    fact_size += factor_sizes[i];
  }

  float* d_factor_matrices = nullptr;
  cudaCheckError(cudaMalloc(&d_factor_matrices, sizeof(float) * fact_size));
  idx = 0;
  fact_size = 0;
  for (int i = 0; i < order; ++i) {
    if (i == ncm) continue;
    cudaCheckError(cudaMemcpy(d_factor_matrices + fact_size,
                              factor_matrices[i],
                              sizeof(float) * factor_sizes[i],
                              cudaMemcpyHostToDevice));
    fact_size += factor_sizes[i];
  }

  uint64_t* d_fact_ofst = nullptr;
  cudaCheckError(cudaMalloc(&d_fact_ofst, sizeof(uint64_t) * (order - 1)));
  cudaCheckError(cudaMemcpy(d_fact_ofst, fact_offsets, sizeof(uint64_t) * (order - 1), cudaMemcpyHostToDevice));

  uint64_t* d_ranks = nullptr;
  cudaCheckError(cudaMalloc(&d_ranks, sizeof(uint64_t) * order));
  cudaCheckError(cudaMemcpy(d_ranks, ranks, sizeof(uint64_t) * order, cudaMemcpyHostToDevice));

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
  size_t sharedMemBytes = ranks[2] * ranks[3] * sizeof(float) + ranks[3] * sizeof(float);

  auto start = std::chrono::high_resolution_clock::now();
  for (uint64_t i_ptr = 0; i_ptr < num_i; ++i_ptr) {
    uint64_t begin = mode_ptrs[1][i_ptr];
    uint64_t end = mode_ptrs[1][i_ptr + 1];
    if (begin >= end) continue;
    dim3 gridDim(static_cast<unsigned int>(end - begin));
    if (gridDim.x == 0) continue;
    uint64_t i = mode_idxs[0][i_ptr];
    cudaStream_t stream = streams[i_ptr % desired_streams];
    kernel_ttmc4_tiny_stream<<<gridDim, blockDim, sharedMemBytes, stream>>>(
      d_meta_data, values,
      d_factor_matrices, d_fact_ofst,
      d_arr_O, d_ranks, ncm, order,
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

  cudaFree(d_meta_data);
  cudaFree(d_factor_matrices);
  cudaFree(d_fact_ofst);
  cudaFree(d_ranks);

  return static_cast<double>(duration) / 1000.0;
}

/* End of GPU kernel definitions */
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/* Start of host launch/orchestration function */
void run_ttmc_cuda(
  uint64_t** mode_ptrs, uint64_t** mode_idxs, float* values,
  float** factor_matrices, uint64_t* factor_sizes,
  float* arr_O, uint64_t arr_O_size, int ncm, 
  uint64_t* ranks, int order,
  uint64_t size_mode_ptr[], uint64_t size_mode_idx[], uint64_t* dimensions)
  { 
    uint64_t total_values = size_mode_idx[order - 1];
    std::vector<int> other_modes;
    other_modes.reserve(order - 1);
    for (int mode = 0; mode < order; ++mode) {
      if (mode != ncm) {
        other_modes.push_back(mode);
      }
    }
    if (other_modes.size() < 2) {
      throw std::runtime_error("Tensor order too small for contraction");
    }
    int idx_A = other_modes[0];
    int idx_B = other_modes[1];
    int idx_C = (other_modes.size() > 2) ? other_modes[2] : -1;
    int f1 = ranks[idx_A];
    int f2 = ranks[idx_B];
    int f3 = (idx_C >= 0) ? ranks[idx_C] : 0;
    
    // Allocate device memory
    uint64_t *d_mode_0_idx, *d_mode_1_ptr;
    uint64_t *d_mode_1_idx, *d_mode_2_ptr, *d_mode_2_idx;
    uint64_t *d_mode_3_ptr = nullptr, *d_mode_3_idx = nullptr;
    float *d_values, *d_arr_A, *d_arr_B, *d_arr_C = nullptr, *d_arr_O;
    
    // cudaMalloc(&d_mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr);
    cudaMalloc(&d_mode_0_idx, sizeof(uint64_t) * size_mode_idx[0]);
    cudaMalloc(&d_mode_1_ptr, sizeof(uint64_t) * size_mode_ptr[1]);
    cudaMalloc(&d_mode_1_idx, sizeof(uint64_t) * size_mode_idx[1]);
    cudaMalloc(&d_mode_2_ptr, sizeof(uint64_t) * size_mode_ptr[2]);
    cudaMalloc(&d_mode_2_idx, sizeof(uint64_t) * size_mode_idx[2]);
    if (order >= 4) {
      cudaMalloc(&d_mode_3_ptr, sizeof(uint64_t) * size_mode_ptr[3]);
      cudaMalloc(&d_mode_3_idx, sizeof(uint64_t) * size_mode_idx[3]);
    }
    cudaMalloc(&d_values, sizeof(float) * total_values);
    cudaMalloc(&d_arr_A, sizeof(float) * factor_sizes[idx_A]);
    cudaMalloc(&d_arr_B, sizeof(float) * factor_sizes[idx_B]);
    if (idx_C >= 0) {
      cudaMalloc(&d_arr_C, sizeof(float) * factor_sizes[idx_C]);
    }
    cudaMalloc(&d_arr_O, sizeof(float) * arr_O_size);
  
    // Copy data to device
    // cudaMemcpy(d_mode_0_ptr, mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_0_idx, mode_idxs[0], sizeof(uint64_t) * size_mode_idx[0], cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_1_ptr, mode_ptrs[1], sizeof(uint64_t) * size_mode_ptr[1], cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_1_idx, mode_idxs[1], sizeof(uint64_t) * size_mode_idx[1], cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_2_ptr, mode_ptrs[2], sizeof(uint64_t) * size_mode_ptr[2], cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_2_idx, mode_idxs[2], sizeof(uint64_t) * size_mode_idx[2], cudaMemcpyHostToDevice);
    if (order >= 4) {
      cudaMemcpy(d_mode_3_ptr, mode_ptrs[3], sizeof(uint64_t) * size_mode_ptr[3], cudaMemcpyHostToDevice);
      cudaMemcpy(d_mode_3_idx, mode_idxs[3], sizeof(uint64_t) * size_mode_idx[3], cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_values, values, sizeof(float) * total_values, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_A, factor_matrices[idx_A], sizeof(float) * factor_sizes[idx_A], cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_B, factor_matrices[idx_B], sizeof(float) * factor_sizes[idx_B], cudaMemcpyHostToDevice);
    if (idx_C >= 0) {
      cudaMemcpy(d_arr_C, factor_matrices[idx_C], sizeof(float) * factor_sizes[idx_C], cudaMemcpyHostToDevice);
    }
    // cudaMemcpy(d_arr_O, arr_O, sizeof(float) * arr_O_size, cudaMemcpyHostToDevice);
    cudaMemset(d_arr_O, 0, sizeof(float) * arr_O_size);
    
    
    switch (ncm) {
      case 0: {
      if (order == 3) {
        bool force_static = getEnvFlag("TTMC_FORCE_STATIC");
        bool force_dynamic = getEnvFlag("TTMC_FORCE_DYNAMIC");
        bool analyze = getEnvFlag("TTMC_ANALYZE");

        uint64_t num_i = size_mode_ptr[1] - 1;
        FiberStats stats = analyzeFiberStats(mode_ptrs[1], mode_ptrs[2], num_i);
        if (analyze) {
          std::cout << "TTMC_ANALYZE: k_per_j min=" << stats.min_k_per_j
                    << " max=" << stats.max_k_per_j
                    << " avg=" << stats.avg_k_per_j
                    << std::endl;
        }
        DynamicHints hints = chooseDynamicHints(stats.avg_k_per_j, force_dynamic);
        uint64_t base_tile = hints.base_tile;
        uint64_t k_tile = hints.k_tile;
        int dynamic_block_hint = hints.dynamic_block_hint;
        int grid_factor_hint = hints.grid_factor_hint;

        // 3D selector policy:
        // - ultra_tiny_k: freebase-like (avg~1, max<=5) -> static block-per-i.
        // - long_tail_sparse: delicious/flickr-like (very long max fibers, low avg) -> static block-per-i.
        // - medium_fiber_dynamic_tuned: mid avg fibers, modest max -> dynamic with tuned launch hints.
        bool ultra_tiny_k = (stats.max_k_per_j <= 5 && stats.avg_k_per_j <= 1.10);
        bool long_tail_sparse =
          (stats.avg_k_per_j <= 12.0 &&
           stats.max_k_per_j >= 15000 &&
           dimensions[0] <= 700000);
        bool medium_fiber_dynamic_tuned =
          (stats.avg_k_per_j >= 300.0 &&
           stats.avg_k_per_j <= 1500.0 &&
           stats.max_k_per_j <= 2000 &&
           num_i <= 512);

        if (!force_dynamic && !force_static && medium_fiber_dynamic_tuned) {
          if (k_tile < 512) k_tile = 512;
          dynamic_block_hint = 128;
          if (grid_factor_hint < 12) grid_factor_hint = 12;
          base_tile = 64;
        }

        bool prefer_static = force_static || (!force_dynamic && (ultra_tiny_k || long_tail_sparse));
        if (f1 > 64) prefer_static = false;

        int device_id = 0;
        cudaCheckError(cudaGetDevice(&device_id));
        cudaDeviceProp prop{};
        cudaCheckError(cudaGetDeviceProperties(&prop, device_id));

        if (prefer_static)
        {
          int default_block = (ultra_tiny_k || long_tail_sparse) ? 768 : 1024;
          int block_size = getEnvInt("TTMC_BLOCK_PER_I_BLOCK", default_block);
          if (block_size < 32) block_size = 32;
          block_size = (block_size / 32) * 32;
          if (block_size == 0) block_size = 32;
          int num_warps = block_size / 32;
          if (num_warps <= 0) num_warps = 1;
          size_t sharedMemBytes =
            static_cast<size_t>(num_warps) * f2 * sizeof(float) +
            static_cast<size_t>(f1) * f2 * sizeof(float);
          int grid_size = static_cast<int>(size_mode_idx[0]);

          auto start = std::chrono::high_resolution_clock::now();
          kernel_ttmc3_static_block_per_i<<<grid_size, block_size, sharedMemBytes>>>(
            d_mode_0_idx,
            d_mode_1_ptr, d_mode_1_idx,
            d_mode_2_ptr, d_mode_2_idx,
            d_values, d_arr_A, d_arr_B, d_arr_O,
            f1, f2, num_warps
          );
          cudaCheckError(cudaGetLastError());
          cudaCheckError(cudaDeviceSynchronize());
          auto end = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
          cout << "Method: kernel_ttmc3_static_block_per_i, Time: " << duration / 1000.0 << " ms" << endl;
        }
        else {
          std::vector<TaskRange> host_tasks;
          size_t reserve_hint = computeTaskReserveHint(size_mode_idx[1]);
          buildDynamicTaskRanges(mode_ptrs[1], mode_ptrs[2], num_i, base_tile, k_tile, host_tasks, reserve_hint);

          if (std::getenv("TTMC_DEBUG_TASKS")) {
            std::cout << "Generated GPU tasks: " << host_tasks.size() << std::endl;
          }

          TaskRange* d_tasks = nullptr;
          unsigned long long* d_task_counter = nullptr;
          allocateTaskBuffers(host_tasks, &d_tasks, &d_task_counter);

          int block_size = dynamic_block_hint;
          const int warp_size = 32;
          int warps_per_block = block_size / warp_size;
          const bool use_register_accum = (f1 <= 64 && f2 <= warp_size);
          size_t sharedMemBytes = 0;
          if (!use_register_accum) {
            sharedMemBytes = (size_t)warps_per_block * f2 * sizeof(float) + (size_t)warps_per_block * f1 * f2 * sizeof(float);
          }
          const size_t default_shared = prop.sharedMemPerBlock;
          size_t max_shared = default_shared;
          if (prop.sharedMemPerBlockOptin > default_shared) {
            max_shared = prop.sharedMemPerBlockOptin;
          }
          while (!use_register_accum && sharedMemBytes > max_shared && block_size > warp_size) {
            block_size -= warp_size;
            warps_per_block = block_size / warp_size;
            sharedMemBytes = (size_t)warps_per_block * f2 * sizeof(float) + (size_t)warps_per_block * f1 * f2 * sizeof(float);
          }
          if (warps_per_block == 0) {
            block_size = warp_size;
            warps_per_block = 1;
            if (!use_register_accum) {
              sharedMemBytes = (size_t)warps_per_block * f2 * sizeof(float) + (size_t)warps_per_block * f1 * f2 * sizeof(float);
            }
            else {
              sharedMemBytes = 0;
            }
          }
          if (sharedMemBytes > max_shared) {
            sharedMemBytes = max_shared;
          }
          if (sharedMemBytes > default_shared && prop.sharedMemPerBlockOptin > default_shared) {
            int requested = static_cast<int>(std::min(sharedMemBytes, max_shared));
            cudaCheckError(cudaFuncSetAttribute(
              kernel_ttmc3_dynamic_tasks,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              requested));
          }
          else {
            cudaCheckError(cudaFuncSetAttribute(
              kernel_ttmc3_dynamic_tasks,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              static_cast<int>(default_shared)));
          }
          int grid_size = std::max(prop.multiProcessorCount * grid_factor_hint, 1);

          if (std::getenv("TTMC_DEBUG_LAUNCH")) {
            std::cout << "TTMC_DEBUG_LAUNCH: grid=" << grid_size
                      << " block=" << block_size
                      << " shared=" << sharedMemBytes
                      << " tasks=" << host_tasks.size()
                      << std::endl;
          }

          auto start = std::chrono::high_resolution_clock::now();
          if (!host_tasks.empty()) {
            kernel_ttmc3_dynamic_tasks<<<grid_size, block_size, sharedMemBytes>>>(
              d_mode_0_idx,
              d_mode_1_ptr, d_mode_1_idx,
              d_mode_2_ptr, d_mode_2_idx,
              d_values, d_arr_A, d_arr_B, d_arr_O, f1, f2,
              d_tasks, static_cast<uint64_t>(host_tasks.size()), d_task_counter
            );
          }
          cudaCheckError(cudaGetLastError());
          cudaCheckError(cudaDeviceSynchronize());
          auto end = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
          cout << "Method: kernel_ttmc3_dynamic_tasks, Time: " << duration / 1000.0 << " ms" << endl;

          freeTaskBuffers(d_tasks, d_task_counter);
        }
      }
      else if (order == 4) {
        bool force_dynamic = getEnvFlag("TTMC_FORCE_DYNAMIC");
        bool analyze = getEnvFlag("TTMC_ANALYZE");

        uint64_t num_i = size_mode_ptr[1] - 1;
        FiberStats stats = analyzeFiberStats(mode_ptrs[1], mode_ptrs[2], num_i);
        double avg_k_per_j = stats.avg_k_per_j;
        uint64_t max_dim = 0;
        for (int dim = 0; dim < order; ++dim) {
          max_dim = std::max<uint64_t>(max_dim, dimensions[dim]);
        }

        bool force_tiny_streams = getEnvFlag("TTMC_FORCE_TINY_STREAMS");
        bool disable_tiny_streams = getEnvFlag("TTMC_DISABLE_TINY_STREAMS");
        bool debug_tiny = getEnvFlag("TTMC_DEBUG_TINY4D");
        int tiny_dim_env = getEnvInt("TTMC_TINY4D_MAX_DIM", 500000);
        if (tiny_dim_env <= 0) tiny_dim_env = 500000;
        int tiny_nnz_env = getEnvInt("TTMC_TINY4D_MAX_NNZ", 60000000);
        if (tiny_nnz_env <= 0) tiny_nnz_env = 60000000;
        int tiny_avg_env = getEnvInt("TTMC_TINY4D_MAX_AVG_K", 64);
        if (tiny_avg_env <= 0) tiny_avg_env = 64;
        int tiny_small_dim_env = getEnvInt("TTMC_TINY4D_SMALL_DIM", 20000);
        if (tiny_small_dim_env <= 0) tiny_small_dim_env = 20000;
        int tiny_small_nnz_env = getEnvInt("TTMC_TINY4D_SMALL_NNZ", 10000000);
        if (tiny_small_nnz_env <= 0) tiny_small_nnz_env = 10000000;

        bool prefer_tiny_streams = false;
        if (idx_C >= 0 && d_mode_3_ptr != nullptr && d_arr_C != nullptr) {
          if (force_tiny_streams) {
            prefer_tiny_streams = true;
          }
          else if (!force_dynamic && !disable_tiny_streams) {
            bool under_dim = max_dim <= static_cast<uint64_t>(tiny_dim_env);
            bool under_nnz = total_values <= static_cast<uint64_t>(tiny_nnz_env);
            bool under_avg = avg_k_per_j <= static_cast<double>(tiny_avg_env);
            bool very_small = max_dim <= static_cast<uint64_t>(tiny_small_dim_env) &&
                              total_values <= static_cast<uint64_t>(tiny_small_nnz_env);
            // For compact tensors (e.g., uber), stream microkernel can still win even when avg_k is high.
            prefer_tiny_streams = under_dim && under_nnz && (under_avg || very_small);
          }
        }

        if (analyze) {
          std::cout << "TTMC_ANALYZE: k_per_j min=" << stats.min_k_per_j
                    << " max=" << stats.max_k_per_j
                    << " avg=" << avg_k_per_j
                    << std::endl;
        }

        if (debug_tiny) {
          std::cout << "TTMC_DEBUG_TINY4D: avg_k=" << avg_k_per_j
                    << " nnz=" << total_values
                    << " max_dim=" << max_dim
                    << " small_dim_th=" << tiny_small_dim_env
                    << " small_nnz_th=" << tiny_small_nnz_env
                    << " prefer=" << (prefer_tiny_streams ? "yes" : "no")
                    << std::endl;
        }

        if (prefer_tiny_streams) {
          int default_streams = static_cast<int>(std::min<uint64_t>(num_i, static_cast<uint64_t>(32)));
          if (default_streams <= 0) default_streams = 1;
          int stream_hint = getEnvInt("TTMC_TINY4D_NUM_STREAMS", default_streams);
          if (stream_hint <= 0) stream_hint = default_streams;
          double duration_ms = launch_ttmc4_tiny_streams_flat(
            mode_ptrs, mode_idxs,
            d_values,
            factor_matrices, factor_sizes,
            d_arr_O,
            ncm, ranks, order,
            size_mode_ptr, size_mode_idx,
            dimensions,
            stream_hint
          );
          cout << "Method: launch_ttmc4_tiny_streams_flat/kernel_ttmc4_tiny_stream, Time: " << duration_ms << " ms" << endl;
        }
        else {
          DynamicHints hints = chooseDynamicHints(avg_k_per_j, force_dynamic);
          uint64_t base_tile = hints.base_tile;
          uint64_t k_tile = hints.k_tile;
          int dynamic_block_hint = hints.dynamic_block_hint;
          int grid_factor_hint = hints.grid_factor_hint;

          int device_id = 0;
          cudaCheckError(cudaGetDevice(&device_id));
          cudaDeviceProp prop{};
          cudaCheckError(cudaGetDeviceProperties(&prop, device_id));

          std::vector<TaskRange> host_tasks;
          size_t reserve_hint = computeTaskReserveHint(size_mode_idx[1]);
          buildDynamicTaskRanges(mode_ptrs[1], mode_ptrs[2], num_i, base_tile, k_tile, host_tasks, reserve_hint);

          if (std::getenv("TTMC_DEBUG_TASKS")) {
            std::cout << "Generated GPU tasks (4D): " << host_tasks.size() << std::endl;
          }

          TaskRange* d_tasks = nullptr;
          unsigned long long* d_task_counter = nullptr;
          allocateTaskBuffers(host_tasks, &d_tasks, &d_task_counter);

          int block_size = dynamic_block_hint;
          const int warp_size = 32;
          int warps_per_block = block_size / warp_size;
          size_t sharedMemBytes = (size_t)warps_per_block * ((size_t)f2 * f3 + f2) * sizeof(float);
          const size_t default_shared = prop.sharedMemPerBlock;
          size_t max_shared = (prop.sharedMemPerBlockOptin > default_shared) ? prop.sharedMemPerBlockOptin : default_shared;

          while (sharedMemBytes > max_shared && block_size > warp_size) {
            block_size -= warp_size;
            warps_per_block = block_size / warp_size;
            sharedMemBytes = (size_t)warps_per_block * ((size_t)f2 * f3 + f2) * sizeof(float);
          }
          if (warps_per_block == 0) {
            block_size = warp_size;
            warps_per_block = 1;
            sharedMemBytes = (size_t)warps_per_block * ((size_t)f2 * f3 + f2) * sizeof(float);
          }
          if (sharedMemBytes > max_shared) {
            sharedMemBytes = max_shared;
          }
          cudaCheckError(cudaFuncSetAttribute(
            kernel_ttmc4_dynamic_tasks,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(std::min(sharedMemBytes, max_shared))));

          int grid_size = std::max(prop.multiProcessorCount * grid_factor_hint, 1);
          if (std::getenv("TTMC_DEBUG_LAUNCH")) {
            std::cout << "TTMC_DEBUG_LAUNCH_4D: grid=" << grid_size
                      << " block=" << block_size
                      << " shared=" << sharedMemBytes
                      << " tasks=" << host_tasks.size()
                      << std::endl;
          }

          auto start = std::chrono::high_resolution_clock::now();
          if (!host_tasks.empty()) {
            kernel_ttmc4_dynamic_tasks<<<grid_size, block_size, sharedMemBytes>>>(
              d_mode_0_idx,
              d_mode_1_ptr, d_mode_1_idx,
              d_mode_2_ptr, d_mode_2_idx,
              d_mode_3_ptr, d_mode_3_idx,
              d_values,
              d_arr_A, d_arr_B, d_arr_C,
              d_arr_O,
              f1, f2, f3,
              d_tasks, static_cast<uint64_t>(host_tasks.size()), d_task_counter
            );
          }
          cudaCheckError(cudaGetLastError());
          cudaCheckError(cudaDeviceSynchronize());
          auto end = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
          cout << "Method: kernel_ttmc4_dynamic_tasks, Time: " << duration / 1000.0 << " ms" << endl;

          freeTaskBuffers(d_tasks, d_task_counter);
        }
        }
        break;
      }
      default:
        throw std::runtime_error("Unsupported ncm in run_ttmc_cuda. Only ncm=0 is implemented.");
    }
  

    cudaDeviceSynchronize();
    // Copy results back to host
    cudaMemcpy(arr_O, d_arr_O, sizeof(float) * arr_O_size, cudaMemcpyDeviceToHost);
  
    // Free device memory
    // cudaFree(d_mode_0_ptr);
    cudaFree(d_mode_0_idx);
    cudaFree(d_mode_1_ptr);
    cudaFree(d_mode_1_idx);
    cudaFree(d_mode_2_ptr);
    cudaFree(d_mode_2_idx);
    if (d_mode_3_ptr) cudaFree(d_mode_3_ptr);
    if (d_mode_3_idx) cudaFree(d_mode_3_idx);
    cudaFree(d_values);
    cudaFree(d_arr_A);
    cudaFree(d_arr_B);
    if (d_arr_C) cudaFree(d_arr_C);
    cudaFree(d_arr_O);
  
  }
/* End of host launch/orchestration function */
////////////////////////////////////////////////////////////////////

// Include the reference implementation for validation
#define INCLUDED_AS_LIBRARY
#include "v2_cpu_factorize_n_fuse.cu"

static bool isUnsignedIntegerToken(const char* token)
{
  if (!token || *token == '\0') return false;
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(token); *p != '\0'; ++p) {
    if (!std::isdigit(*p)) return false;
  }
  return true;
}

int main(int argc, char* argv[])
{
  bool verbose = false;
  string csf_file;
  std::vector<uint64_t> ranks;
  int ncm = 0;
  bool verify = false;  // Default: don't verify results
  
  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
      string arg = argv[i];
      if (arg == "-v" || arg == "--verbose") {
          verbose = true;
      }
      else if ((arg == "-r" || arg == "--ranks") && i + 1 < argc) {
          // Collect numeric rank tokens only.
          while (i + 1 < argc && isUnsignedIntegerToken(argv[i + 1])) {
              ranks.push_back(static_cast<uint64_t>(atoi(argv[++i])));
          }
      }
      else if ((arg == "-n" || arg == "--ncm") && i + 1 < argc) {
          ncm = atoi(argv[++i]);
      }
      else if (arg == "--verify") {
          verify = true;
      }
      else if (csf_file.empty()) {
          csf_file = arg;
      }
  }
  
  if (csf_file.empty()) {
      cerr << "Usage: " << argv[0] << " [options] <csf_file>" << endl;
      cerr << "Options:" << endl;
      cerr << "  -v, --verbose      Enable verbose output" << endl;
      cerr << "  -r, --ranks <r1> [r2 ...]  Set all factor matrix ranks (space separated)" << endl;
      cerr << "  -n, --ncm <mode>   Contraction mode (only 0 is supported here, default 0)" << endl;
      cerr << "  --verify           Verify results against reference implementation" << endl;
      return 1;
  }
  if (ncm != 0) {
      cerr << "Error: ncm=" << ncm << " is not supported in this unified kernel. Use -n 0." << endl;
      return 1;
  }
  
  try {
      // Load the CSF tensor
      CSFTensor tensor = readCSFTensor(csf_file);
      
      if (verbose) {
          cout << "Loaded tensor from " << csf_file << endl;
          
          cout << "Nonzeros: " << tensor.values.size() << endl;
      }
      
      // Convert CSF tensor to arrays (N-dimensional, zero-copy)
      std::vector<uint64_t*> mode_ptrs, mode_idxs;
      float* values;
      int order;
      getCSFArrays(tensor, mode_ptrs, mode_idxs, values, order);
      
      // Check that number of ranks matches tensor order
      if (ranks.size() < static_cast<size_t>(order)) {
          cerr << "Error: Number of ranks (" << ranks.size() << ") does not match tensor order (" << order << ")." << endl;
          return 1;
      }
      std::vector<size_t> size_mode_ptr(order), size_mode_idx(order);
      for (int i = 0; i < order; ++i) {
          size_mode_ptr[i] = tensor.ptrs[i].size();
          size_mode_idx[i] = tensor.idxs[i].size();
      }

      // Generate 'order' number of factor matrices
      std::vector<float*> factor_matrices(order, nullptr);
      std::vector<uint64_t> factor_sizes(order);
      for (int i = 0; i < order; ++i) {
        if(i != ncm){
          generate_matrix(tensor.dimensions[i], ranks[i], 42 + i, factor_matrices[i]);
        }
        factor_sizes[i] = tensor.dimensions[i] * ranks[i];
      }

      if (verbose) {
        for (int i = 0; i < order; ++i) {
            cout << "Factor matrix " << i << ": " << tensor.dimensions[i] << " x " << ranks[i] << endl;
        }
      }
      // Output tensor: 
      uint64_t arr_O_size = 1;
      for (int i = 0; i < order; ++i){
        if(i != ncm) arr_O_size *= ranks[i];
        else arr_O_size *= tensor.dimensions[i];
      }
       
      float* arr_O = allocate_aligned_array(arr_O_size);
      float* ref_O = nullptr;
      if (verify) {
          ref_O = allocate_aligned_array(arr_O_size);
      }
      
      // Run GPU implementation (run_ttmc_cuda) first
      if (verbose) {
          cout << "Running run_ttmc_cuda implementation..." << endl;
      }
      auto start = std::chrono::high_resolution_clock::now();
      
      run_ttmc_cuda(
          mode_ptrs.data(), mode_idxs.data(), values,
          factor_matrices.data(), factor_sizes.data(),
          arr_O, arr_O_size,
          ncm, ranks.data(), order,
          size_mode_ptr.data(), size_mode_idx.data(), tensor.dimensions.data()
      );
      
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      
      bool valid = true;
      float ref_duration = 0.0;
      
      if (verify) {
          // Only run reference implementation and validate if requested
          if (verbose) {
            cout << "Running reference implementation (cpu_factorize_n_fuse)..." << endl;
          }
          auto ref_start = std::chrono::high_resolution_clock::now();
          
          cpu_factorize_n_fuse(
              mode_ptrs.data(), mode_idxs.data(),
              values, factor_matrices.data(), ref_O,
              ncm, ranks.data(), order, tensor.dimensions.data() 
          );
          
          auto ref_end = std::chrono::high_resolution_clock::now();
          ref_duration = std::chrono::duration_cast<std::chrono::microseconds>(ref_end - ref_start).count();
          
          // Validate results using compare_results from matrix_utils.h
          valid = compare_results(arr_O, ref_O, arr_O_size);
          cout << "validation: " << (valid ? "PASSED" : "FAILED") << endl;
      }
      
      // Report results
      if (verbose) {
          cout << "run_ttmc_cuda execution time: " << duration / 1000.0 << " ms" << endl;
          if (verify) {
              cout << "Reference execution time: " << ref_duration / 1000.0 << " ms" << endl;
              cout << "Speedup over reference: " << (float)ref_duration / duration << "x" << endl;
              cout << "Result validation: " << (valid ? "PASSED" : "FAILED") << endl;
          }
      }
      else {
          if (verify) { 
              cout << "Method: run_ttmc_cuda, Time: " << duration / 1000.0 << " ms, Validation: " << (valid ? "PASSED" : "FAILED") << endl;
          }
          else {
              cout << "Method: run_ttmc_cuda, Time: " << duration / 1000.0 << " ms" << endl;
          }
      }
      
      // Clean up
      for(int i = 0; i < order; i++){
        free(factor_matrices[i]);
      }
      free(arr_O);
      if (ref_O) free(ref_O);
      
      return valid ? 0 : 1;
  }
  catch (const std::exception& e) {
      cerr << "Error: " << e.what() << endl;
      return 1;
  }
} 
