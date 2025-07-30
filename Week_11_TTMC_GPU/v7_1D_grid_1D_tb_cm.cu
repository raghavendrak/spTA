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
/*Start of device function for GPU 4 loop Method using COALESCED MEMORY*/
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

/*End of device function for GPU 4 loop Method using COALESCED MEMORY*/
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/*Start of host function for GPU 4 loop Method using COALESCED MEMORY*/
void GPU_4L_CM_host_func(
  uint64_t** mode_ptrs, uint64_t** mode_idxs,
  float* values, float* arr_A, float* arr_B,  
  float* arr_O, uint64_t arr_A_size, uint64_t arr_B_size, uint64_t arr_O_size, int contraction, 
  uint64_t l, uint64_t m, uint64_t n, uint32_t f1, uint32_t f2, uint64_t total_values,
  uint64_t size_mode_ptr[], uint64_t size_mode_idx[])
  { 
    // uint64_t* mode_0_ptr = mode_ptrs[0];
    uint64_t* mode_1_ptr = mode_ptrs[1];
    uint64_t* mode_2_ptr = mode_ptrs[2];
    uint64_t* mode_0_idx = mode_idxs[0];
    uint64_t* mode_1_idx = mode_idxs[1];
    uint64_t* mode_2_idx = mode_idxs[2];

    // uint64_t size_mode_0_ptr = size_mode_ptr[0];
    uint64_t size_mode_1_ptr = size_mode_ptr[1];
    uint64_t size_mode_2_ptr = size_mode_ptr[2];
    uint64_t size_mode_0_idx = size_mode_idx[0];
    uint64_t size_mode_1_idx = size_mode_idx[1];
    uint64_t size_mode_2_idx = size_mode_idx[2];
    
    // Allocate device memory
    uint64_t *d_mode_0_idx, *d_mode_1_ptr;
    uint64_t *d_mode_1_idx, *d_mode_2_ptr, *d_mode_2_idx;
    float *d_values, *d_arr_A, *d_arr_B, *d_arr_O;
    
    // cudaMalloc(&d_mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr);
    cudaMalloc(&d_mode_0_idx, sizeof(uint64_t) * size_mode_0_idx);
    cudaMalloc(&d_mode_1_ptr, sizeof(uint64_t) * size_mode_1_ptr);
    cudaMalloc(&d_mode_1_idx, sizeof(uint64_t) * size_mode_1_idx);
    cudaMalloc(&d_mode_2_ptr, sizeof(uint64_t) * size_mode_2_ptr);
    cudaMalloc(&d_mode_2_idx, sizeof(uint64_t) * size_mode_2_idx);
    cudaMalloc(&d_values, sizeof(float) * total_values);
    cudaMalloc(&d_arr_A, sizeof(float) * arr_A_size);
    cudaMalloc(&d_arr_B, sizeof(float) * arr_B_size);
    cudaMalloc(&d_arr_O, sizeof(float) * arr_O_size);
  
    // Copy data to device
    // cudaMemcpy(d_mode_0_ptr, mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_0_idx, mode_0_idx, sizeof(uint64_t) * size_mode_0_idx, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_1_ptr, mode_1_ptr, sizeof(uint64_t) * size_mode_1_ptr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_1_idx, mode_1_idx, sizeof(uint64_t) * size_mode_1_idx, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_2_ptr, mode_2_ptr, sizeof(uint64_t) * size_mode_2_ptr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_2_idx, mode_2_idx, sizeof(uint64_t) * size_mode_2_idx, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, sizeof(float) * total_values, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_A, arr_A, sizeof(float) * arr_A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_B, arr_B, sizeof(float) * arr_B_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_arr_O, arr_O, sizeof(float) * arr_O_size, cudaMemcpyHostToDevice);
    cudaMemset(d_arr_O, 0, sizeof(float) * arr_O_size);
    
    
    if (contraction == 0) {
    
      // dim3 gridDim(size_mode_0_idx);
      int grid_size = size_mode_0_idx;
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
    else if (contraction == 1) {
    
      // dim3 gridDim(size_mode_0_idx);
      int grid_size = size_mode_0_idx;
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
      cout << "Method: GPU_4L_CM, Time: " << duration / 1000.0 << " ms" << endl;
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
/*End of host function for GPU 4 loop Method using COALESCED MEMORY*/
////////////////////////////////////////////////////////////////////

// Include the reference implementation for validation
#define INCLUDED_AS_LIBRARY
#include "v2_cpu_4loop.cu"

int main(int argc, char* argv[]) {
    bool verbose = false;
    string csf_file;
    uint32_t rank1 = 30, rank2 = 30;
    int ncm = 0;
    bool verify = false;  // Default: don't verify results
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-r1" && i + 1 < argc) {
            rank1 = atoi(argv[++i]);
        } else if (arg == "-r2" && i + 1 < argc) {
            rank2 = atoi(argv[++i]);
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
        cerr << "  -r1 <rank>         Set first factor matrix rank (default 30)" << endl;
        cerr << "  -r2 <rank>         Set second factor matrix rank (default 30)" << endl;
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
        
        std::vector<size_t> size_mode_ptr(order), size_mode_idx(order);
        for (int i = 0; i < order; ++i) {
            size_mode_ptr[i] = tensor.ptrs[i].size();
            size_mode_idx[i] = tensor.idxs[i].size();
        }
        size_t total_values = tensor.values.size();

        if (verbose) {
            for (int i = 0; i < order; ++i) {
                cout << "size_mode_" << i << "_ptr = " << size_mode_ptr[i] << "\n";
                cout << "size_mode_" << i << "_idx = " << size_mode_idx[i] << "\n";
            }
            cout << "total_values    = " << total_values << endl;
        }
        
        vector<uint64_t> dimensions(tensor.order);
        for(int i = 0; i < tensor.order; i++){
            dimensions[i] = tensor.dimensions[i];
        }

        
        // Calculate matrix dimensions based on contraction mode
        uint64_t matrix_dim1 = getMatrixDim1(dimensions, ncm);
        uint64_t matrix_dim2 = getMatrixDim2(dimensions, ncm);
        uint64_t out_dim1 = getOutputDim1(dimensions, ncm);
        
        // Generate factor matrices
        float *arr_A = nullptr, *arr_B = nullptr;
        generate_matrix(matrix_dim1, rank1, 42, arr_A);
        generate_matrix(matrix_dim2, rank2, 43, arr_B);
        
        // Prepare output matrix dimensions
        uint64_t out_dim2 = rank1 * rank2;
        uint64_t arr_A_size = matrix_dim1 * rank1;
        uint64_t arr_B_size = matrix_dim2 * rank2;
        uint64_t arr_O_size = out_dim1 * out_dim2;
        
        if (verbose) {
          cout << "Matrix A dimensions: " << matrix_dim1 << " x " << rank1 << endl;
          cout << "Matrix B dimensions: " << matrix_dim2 << " x " << rank2 << endl;
          cout << "Output dimensions: " << out_dim1 << " x " << rank1 << " x " << rank2 << endl;
        }
        
        // Allocate output array
        float* arr_O = allocate_aligned_array(arr_O_size);
        float* ref_O = nullptr;
        
        if (verify) {
            // Only allocate reference array if verification is needed
            ref_O = allocate_aligned_array(arr_O_size);
        }
        
        // Run this implementation (GPU 4-loop with streams) first
        if (verbose) {
          cout << "Running GPU 4-loop with coalescing memory implementation..." << endl;
        }
        auto start = std::chrono::high_resolution_clock::now();
      
        GPU_4L_CM_host_func(
          mode_ptrs.data(), mode_idxs.data(),
          values,
          arr_A, arr_B, arr_O,
          arr_A_size, arr_B_size, arr_O_size,
          ncm, dimensions[0], dimensions[1], dimensions[2], rank1, rank2,
          total_values,
          size_mode_ptr.data(), size_mode_idx.data()
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        bool valid = true;
        float ref_duration = 0.0;
        
        if (verify) {
            // Only run reference implementation and validate if requested
            if (verbose) {
                cout << "Running reference implementation (CPU 4-loop)..." << endl;
            }
            auto ref_start = std::chrono::high_resolution_clock::now();
            uint64_t rank1_64 = rank1;
            uint64_t rank2_64 = rank2;
            performContraction_cpu_2(
              mode_ptrs.data(), mode_idxs.data(),
              values, arr_A, arr_B, ref_O,
              arr_A_size, arr_B_size, arr_O_size, ncm,
              dimensions[0], dimensions[1], dimensions[2], rank1_64, rank2_64
            );
            
            auto ref_end = std::chrono::high_resolution_clock::now();
            ref_duration = std::chrono::duration_cast<std::chrono::microseconds>(ref_end - ref_start).count();
            
            // Validate results using compare_results from matrix_utils.h
            valid = compare_results(arr_O, ref_O, arr_O_size);
            cout << "validation: " << (valid ? "PASSED" : "FAILED") << endl;  
        }
        
        // Report results
        if(verbose){
          cout << "Method: GPU_4L_CM, Time: " << duration / 1000.0 << " ms" << endl;
        }
        
        // Clean up
        // for (int i = 0; i < tensor.order; ++i) {
        //     delete[] mode_ptrs[i];
        //     delete[] mode_idxs[i];
        // }
        // delete[] values;
        delete[] arr_A;
        delete[] arr_B;
        free(arr_O);
        if (ref_O) free(ref_O);
        
        return valid ? 0 : 1;
    }
    catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}