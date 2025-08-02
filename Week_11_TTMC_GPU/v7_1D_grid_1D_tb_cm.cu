#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <cuda_runtime.h>
#include "csf_tensor.h"
#include "matrix_utils.h"

using namespace std;

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

/////////////////////////////////////////////////////////////////////
/*Start of device function for GPU 1D-grid-1D-tb-cm Method*/
__global__ void GPU_4L_CM_device_func_ncm_0( 
  const uint64_t* __restrict__ mode_0_idx,
  const uint64_t* __restrict__ mode_1_ptr, const uint64_t* __restrict__ mode_1_idx,
  const uint64_t* __restrict__ mode_2_ptr, const uint64_t* __restrict__ mode_2_idx,
  const float* __restrict__ values, float* arr_A,  float* arr_B,  float* arr_O,
  uint32_t f1, uint32_t f2,  int num_warps)
{
  extern __shared__ float buf[];
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

  for(uint64_t j_ptr_offset = mode_1_ptr[i_ptr]; j_ptr_offset < mode_1_ptr[i_ptr + 1]; j_ptr_offset += num_warps){
    uint64_t j_ptr =  j_ptr_offset + warp_id;
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
  }
  __syncthreads();
  
  for(uint32_t r_offset = 0; r_offset < f1; r_offset += num_warps){
    uint32_t r = r_offset + warp_id;
    if(r < f1){
      for(uint32_t s_offset = 0; s_offset < f2; s_offset += warp_size){
        uint32_t s = s_offset + tid_in_warp;
        if(s < f2){
          atomicAdd(&arr_O[i * f1* f2 + r * f2 + s], buf[num_warps * f2 + r * f2 + s]);
        }
      }
    }
  }
}

__global__ void GPU_4L_CM_device_func_ncm_1( 
  const uint64_t* __restrict__ mode_0_idx,
  const uint64_t* __restrict__ mode_1_ptr, const uint64_t* __restrict__ mode_1_idx,
  const uint64_t* __restrict__ mode_2_ptr, const uint64_t* __restrict__ mode_2_idx,
  const float* __restrict__ values, float* arr_A,  float* arr_B,  float* arr_O,
  uint32_t f1, uint32_t f2, int num_warps)
{
  extern __shared__ float buf[];
  int buf_index;
  
  uint64_t i_ptr = blockIdx.x;
  uint64_t i =  mode_0_idx[i_ptr];

  uint32_t warp_size = 32;
  uint32_t warp_id = threadIdx.x / warp_size;
  int tid_in_warp = threadIdx.x % warp_size;

  for(uint64_t j_ptr_offset = mode_1_ptr[i_ptr]; j_ptr_offset < mode_1_ptr[i_ptr + 1]; j_ptr_offset += num_warps){
    uint64_t j_ptr =  j_ptr_offset + warp_id;
    if(j_ptr < mode_1_ptr[i_ptr + 1]){
      uint64_t j = mode_1_idx[j_ptr];

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
      // __syncthreads();
      
      for(uint32_t r = 0; r < f1; ++r){
        for(uint32_t s_offset = 0; s_offset < f2; s_offset += warp_size){
          uint32_t s = s_offset + tid_in_warp;
          if(s < f2){
            atomicAdd(&arr_O[j * f1* f2 + r * f2 + s], buf[warp_id * f2 + s] * arr_A[i * f1 + r]);
          }
        }
      }
      
    }
  }
}

/*End of device function for GPU 1D-grid-1D-tb-cm Method*/
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/*Start of host function for GPU 1D-grid-1D-tb-cm Method*/
void gpu_1D_grid_1D_tb_cm(
  uint64_t** mode_ptrs, uint64_t** mode_idxs, float* values,
  float** factor_matrices, uint64_t* factor_sizes,
  float* arr_O, uint64_t arr_O_size, int ncm, 
  uint64_t* ranks, int order,
  uint64_t size_mode_ptr[], uint64_t size_mode_idx[], uint64_t* dimensions)
  { 
    uint64_t total_values = size_mode_idx[2];
    int idx_A, idx_B;
    if(ncm == 0){
      idx_A = 1;
      idx_B = 2;
    }else if(ncm == 1){
      idx_A = 0;
      idx_B = 2;
    }else if(ncm == 2){
      idx_A = 0;
      idx_B = 1;
    }
    int f1 = ranks[idx_A];
    int f2 = ranks[idx_B];
    
    // Allocate device memory
    uint64_t *d_mode_0_idx, *d_mode_1_ptr;
    uint64_t *d_mode_1_idx, *d_mode_2_ptr, *d_mode_2_idx;
    float *d_values, *d_arr_A, *d_arr_B, *d_arr_O;
    
    // cudaMalloc(&d_mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr);
    cudaMalloc(&d_mode_0_idx, sizeof(uint64_t) * size_mode_idx[0]);
    cudaMalloc(&d_mode_1_ptr, sizeof(uint64_t) * size_mode_ptr[1]);
    cudaMalloc(&d_mode_1_idx, sizeof(uint64_t) * size_mode_idx[1]);
    cudaMalloc(&d_mode_2_ptr, sizeof(uint64_t) * size_mode_ptr[2]);
    cudaMalloc(&d_mode_2_idx, sizeof(uint64_t) * size_mode_idx[2]);
    cudaMalloc(&d_values, sizeof(float) * total_values);
    cudaMalloc(&d_arr_A, sizeof(float) * factor_sizes[idx_A]);
    cudaMalloc(&d_arr_B, sizeof(float) * factor_sizes[idx_B]);
    cudaMalloc(&d_arr_O, sizeof(float) * arr_O_size);
  
    // Copy data to device
    // cudaMemcpy(d_mode_0_ptr, mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_0_idx, mode_idxs[0], sizeof(uint64_t) * size_mode_idx[0], cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_1_ptr, mode_ptrs[1], sizeof(uint64_t) * size_mode_ptr[1], cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_1_idx, mode_idxs[1], sizeof(uint64_t) * size_mode_idx[1], cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_2_ptr, mode_ptrs[2], sizeof(uint64_t) * size_mode_ptr[2], cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_2_idx, mode_idxs[2], sizeof(uint64_t) * size_mode_idx[2], cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, sizeof(float) * total_values, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_A, factor_matrices[idx_A], sizeof(float) * factor_sizes[idx_A], cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_B, factor_matrices[idx_B], sizeof(float) * factor_sizes[idx_B], cudaMemcpyHostToDevice);
    // cudaMemcpy(d_arr_O, arr_O, sizeof(float) * arr_O_size, cudaMemcpyHostToDevice);
    cudaMemset(d_arr_O, 0, sizeof(float) * arr_O_size);
    
    
    if (ncm == 0) {
    
      // dim3 gridDim(size_mode_0_idx);
      int grid_size = size_mode_idx[0];
      // dim3 blockDim(1024);
      int block_size = 512, warp_size = 32;
      int num_warps = (block_size + warp_size - 1) / warp_size;
      int sharedMemBytes =  num_warps * f2 * sizeof(float) + f1 * f2 * sizeof(float);
      
      auto start = std::chrono::high_resolution_clock::now();
      GPU_4L_CM_device_func_ncm_0<<<grid_size, block_size, sharedMemBytes>>>(
        d_mode_0_idx,
        d_mode_1_ptr, d_mode_1_idx,
        d_mode_2_ptr, d_mode_2_idx,
        d_values, d_arr_A, d_arr_B, d_arr_O, f1, f2, num_warps
      );
      cudaDeviceSynchronize();
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      cout << "Method: 1D-grid-1D-tb-cm-W, Time: " << duration / 1000.0 << " ms" << endl;
    }
    else if (ncm == 1) {
    
      // dim3 gridDim(size_mode_0_idx);
      int grid_size = size_mode_idx[0];
      // dim3 blockDim(1024);
      int block_size = 1024, warp_size = 32;
      int num_warps = (block_size + warp_size - 1) / warp_size;
      int sharedMemBytes =  num_warps * f2 * sizeof(float);
      
      auto start = std::chrono::high_resolution_clock::now();
      GPU_4L_CM_device_func_ncm_1<<<grid_size, block_size, sharedMemBytes>>>(
        d_mode_0_idx,
        d_mode_1_ptr, d_mode_1_idx,
        d_mode_2_ptr, d_mode_2_idx,
        d_values, d_arr_A, d_arr_B, d_arr_O, f1, f2, num_warps
      );
      cudaDeviceSynchronize();
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      cout << "Method: 1D-grid-1D-tb-cm-W, Time: " << duration / 1000.0 << " ms" << endl;
    }
    /*
    else if(contraction == 2){
      float* buffer_for_ncm_2;
      bool* k_index_buffer;
      
      NUM_STREAMS = 1;
      cout << "No. of streams = " << NUM_STREAMS <<endl;

      cudaMalloc(&buffer_for_ncm_2, n * f2 * NUM_STREAMS * sizeof(float));
      cudaMalloc(&k_index_buffer, n * NUM_STREAMS * sizeof(bool));
      
      // cudaMalloc(&k_indices, n * NUM_STREAMS * sizeof(uint64_t));
      // cudaMalloc(&counter,  NUM_STREAMS * sizeof(uint64_t));
      
      // cudaMemset(buffer_for_ncm_2 , 0, n * f2  * NUM_STREAMS * sizeof(float));
      // cudaMemset(k_index_buffer, 0, n  * NUM_STREAMS * sizeof(bool));

      
      for (uint64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
        i = mode_0_idx[i_ptr];
        j_ptr_offset = mode_1_ptr[i_ptr];
        
        cudaMemset(buffer_for_ncm_2 + n * f2 * (i_ptr % NUM_STREAMS), 0, n * f2  * sizeof(float));
        cudaMemset(k_index_buffer + n * (i_ptr % NUM_STREAMS), 0, n  * sizeof(bool));
        
        dim3 gridDim(mode_1_ptr[i_ptr + 1] - mode_1_ptr[i_ptr]);
        dim3 blockDim(32, 32);

        GPU_4loop_streams_ncm_2_part_1<<<gridDim, blockDim, 0, streams[i_ptr%NUM_STREAMS]>>>(
          d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
          d_values, d_arr_A, d_arr_B, d_arr_O, l, m, n, f1, f2, contraction,
          size_mode_0_ptr, size_mode_1_ptr, size_mode_2_ptr,
          size_mode_0_idx, size_mode_1_idx, size_mode_2_idx,
          i, j_ptr_offset, buffer_for_ncm_2 + n * f2 * (i_ptr % NUM_STREAMS), k_index_buffer + n * (i_ptr % NUM_STREAMS)
        );

        // cudaDeviceSynchronize();
        // pick_non_zero_Ks(k_index_buffer + n * (i_ptr % NUM_STREAMS), k_indices + n * (i_ptr % NUM_STREAMS),  n)

        gridDim.x = n; //TO-DO: have to be optimized
        GPU_4loop_streams_ncm_2_part_2<<<gridDim, blockDim, 0, streams[i_ptr%NUM_STREAMS]>>>(
          d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
          d_values, d_arr_A, d_arr_B, d_arr_O, l, m, n, f1, f2, contraction,
          size_mode_0_ptr, size_mode_1_ptr, size_mode_2_ptr,
          size_mode_0_idx, size_mode_1_idx, size_mode_2_idx,
          i, j_ptr_offset, buffer_for_ncm_2 + n * (i_ptr % NUM_STREAMS), k_index_buffer + n * (i_ptr % NUM_STREAMS)
        );
        cudaGetLastError();  // Check launch err;
        // cudaStreamSynchronize(streams[i_ptr % NUM_STREAMS]);
      }
      
    }
    */
  

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
    cudaFree(d_values);
    cudaFree(d_arr_A);
    cudaFree(d_arr_B);
    cudaFree(d_arr_O);
  
  }
/*End of host function for GPU 1D-grid-1D-tb-cm Method*/
////////////////////////////////////////////////////////////////////

// Include the reference implementation for validation
#define INCLUDED_AS_LIBRARY
#include "v2_cpu_factorize_n_fuse.cu"

int main(int argc, char* argv[]) {
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
      } else if ((arg == "-r" || arg == "--ranks") && i + 1 < argc) {
          // Collect all numbers after -r/--ranks until next arg or end
          while (i + 1 < argc && argv[i + 1][0] != '-') {
              ranks.push_back(static_cast<uint64_t>(atoi(argv[++i])));
          }
      } else if ((arg == "-n" || arg == "--ncm") && i + 1 < argc) {
          ncm = atoi(argv[++i]);
      } else if (arg == "--verify") {
          verify = true;
      } else if (csf_file.empty()) {
          csf_file = arg;
      }
  }
  
  if (csf_file.empty()) {
      cerr << "Usage: " << argv[0] << " [options] <csf_file>" << endl;
      cerr << "Options:" << endl;
      cerr << "  -v, --verbose      Enable verbose output" << endl;
      cerr << "  -r, --ranks <r1> [r2 ...]  Set all factor matrix ranks (space separated)" << endl;
      cerr << "  -n, --ncm <mode>   Set contraction mode (0, 1, or 2, default 0)" << endl;
      cerr << "  --verify           Verify results against reference implementation" << endl;
      return 1;
  }
  
  try {
      // Load the CSF tensor
      CSFTensor tensor = readCSFTensor(csf_file);
      
      if (verbose) {
          cout << "Loaded tensor from " << csf_file << endl;
          cout << "Tensor dimensions: " << tensor.dimensions[0] << " x " << tensor.dimensions[1] << " x " << tensor.dimensions[2] << endl;
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
        generate_matrix(tensor.dimensions[i], ranks[i], 42 + i, factor_matrices[i]);
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
      
      // Run this implementation (GPU 1D grid 1D tb cm) first
      if (verbose) {
          cout << "Running GPU 1D grid 1D tb cm implementation..." << endl;
      }
      auto start = std::chrono::high_resolution_clock::now();
      
      gpu_1D_grid_1D_tb_cm(
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
          cout << "GPU 1D grid 1D tb cm execution time: " << duration / 1000.0 << " ms" << endl;
          if (verify) {
              cout << "Reference execution time: " << ref_duration / 1000.0 << " ms" << endl;
              cout << "Speedup over reference: " << (float)ref_duration / duration << "x" << endl;
              cout << "Result validation: " << (valid ? "PASSED" : "FAILED") << endl;
          }
      } else {
          if (verify) { 
              cout << "Method: 1D_grid_1D_tb_cm, Time: " << duration / 1000.0 << " ms, Validation: " << (valid ? "PASSED" : "FAILED") << endl;
          } else {
              cout << "Method: 1D_grid_1D_tb_cm, Time: " << duration / 1000.0 << " ms" << endl;
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