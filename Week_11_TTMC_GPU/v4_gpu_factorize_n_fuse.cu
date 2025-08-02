#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <chrono>
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
/*Start of device function for GPU 4 loop Method*/
__global__ void contractionKernel_4(
  uint64_t* mode_0_ptr, uint64_t* mode_0_idx,
  uint64_t* mode_1_ptr, uint64_t* mode_1_idx,
  uint64_t* mode_2_ptr, uint64_t* mode_2_idx,
  float* values, float* arr_A, float* arr_B,  
  float* arr_O, uint64_t f1, uint64_t f2, int contraction, float* buffer)
{
  // Compute thread index
  uint64_t j_ptr = blockIdx.x * blockDim.x + threadIdx.x;

  if(contraction == 0 || contraction == 1){
    // Determine the range of valid j_ptr indices
    if (j_ptr < mode_1_ptr[mode_0_ptr[1]]) {
      // Find the corresponding i_ptr for the given j_ptr
      int64_t i_ptr = -1;
      for (uint64_t p = 0; p < mode_0_ptr[1]; ++p) {
        if (j_ptr >= mode_1_ptr[p] && j_ptr < mode_1_ptr[p + 1]) {
          i_ptr = p;
          break;
        }
      }

      // Ensure a valid i_ptr was found
      if (i_ptr == -1) return;

      if(i_ptr >= 0 && i_ptr < mode_0_ptr[1]){
        uint64_t i = mode_0_idx[i_ptr];
        uint64_t j = mode_1_idx[j_ptr];

        // Allocate buffer on a per-thread basis
        //float* buffer = (float*)malloc(f2 * sizeof(float));
        if (buffer == nullptr) {// Handle allocation failure
          if(blockIdx.x == 0 && threadIdx.x == 0){
            printf("Memory allocation failure \n");
            //printf("CUDA Kernel: Memory allocation failure for thread %d-%d. This likely indicates insufficient GPU memory.\n", blockIdx.x, threadIdx.x);
            asm("trap;"); // Force kernel to terminate with error
          }
          return;
        } 

        // Reset buffer
        // memset(buffer, 0, f2 * sizeof(float));

        // Process the k_ptr range associated with j_ptr
        for (uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
          uint64_t k = mode_2_idx[k_ptr];
          float value = values[k_ptr];

          for (uint64_t s = 0; s < f2; ++s) {
            uint64_t index_B = k * f2 + s;
            // atomicAdd(&buffer[s], value * arr_B[index_B]);
            atomicAdd(&buffer[j_ptr * f2 + s], value * arr_B[index_B]);
          }
        }

        // Perform the computation and update `arr_O`
        for (uint64_t r = 0; r < f1; ++r) {
          uint64_t index_A = 0;
          if (contraction == 0){
            index_A = j * f1 + r;
          }
          else if(contraction == 1){
            index_A = i * f1 + r;
          }

          for (uint64_t s = 0; s < f2; ++s) {
            uint64_t index_O = 0;
            if (contraction == 0) {
              index_O = i * f1 * f2 + r * f2 + s;
            } else if (contraction == 1) {
              index_O = j * f1 * f2 + r * f2 + s;
            }
            // atomicAdd(&arr_O[index_O], buffer[s] * arr_A[index_A]);
            atomicAdd(&arr_O[index_O], buffer[j_ptr * f2 + s] * arr_A[index_A]);
          }
        }

        // Free allocated buffer
        // free(buffer);
      }
    }
  }
}

__global__ void contractionKernel_for_second_contraction_part_1(
  uint64_t* mode_0_ptr, uint64_t* mode_0_idx,
  uint64_t* mode_1_ptr, uint64_t* mode_1_idx,
  uint64_t* mode_2_ptr, uint64_t* mode_2_idx,
  float* values, float* arr_A, float* arr_B,  
  float* arr_O, uint64_t n, uint64_t f1, uint64_t f2, int contraction, float* buffer, int* k_buffer)
{
  // Compute thread index
  uint64_t j_ptr = blockIdx.x * blockDim.x + threadIdx.x;

  if (j_ptr < mode_1_ptr[mode_0_ptr[1]]) {
    int64_t i_ptr = -1;
    for (uint64_t idx = 0; idx < mode_0_ptr[1]; ++idx) {
      if (j_ptr >= mode_1_ptr[idx] && j_ptr < mode_1_ptr[idx + 1]) {
        i_ptr = idx;
        break;
      }
    }
    if (i_ptr < 0) return; // Out of bounds check


    if(i_ptr >= 0 && i_ptr < mode_0_ptr[1]){
      // uint64_t i = mode_0_idx[i_ptr];
      uint64_t j = mode_1_idx[j_ptr] ;

      for (uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
        uint64_t k = mode_2_idx[k_ptr];
        atomicAdd(&k_buffer[j_ptr * n + k], 1);
        // k_buffer[j_ptr * n + k] += 1;
        float value = values[k_ptr];

        for (uint64_t s = 0; s < f2; ++s) {
          uint64_t index_B = j * f2 + s;
          uint64_t index_buf = k * f2 + s;

          // if (index_B >= n * f2 || j_ptr * (n * f2) + index_buf >= n * f2 * mode_1_ptr[mode_0_ptr[1]]) {
          //     printf("Out of bound access! \n");
          // }
          atomicAdd(&buffer[j_ptr * (n * f2) + index_buf], value * arr_B[index_B]);
          // buffer[j_ptr * (n * f2) + index_buf] += value * arr_B[index_B];
        }
      }
    }
  }
}


__global__ void contractionKernel_for_second_contraction_part_2(
  uint64_t* mode_0_ptr, uint64_t* mode_0_idx,
  uint64_t* mode_1_ptr, uint64_t* mode_1_idx,
  uint64_t* mode_2_ptr, uint64_t* mode_2_idx,
  float* values, float* arr_A, float* arr_B,  
  float* arr_O, uint64_t n, uint64_t f1, uint64_t f2, int contraction, float* buffer, int* k_buffer)
{
  // Compute thread index
  uint64_t j_ptr = blockIdx.x * blockDim.x + threadIdx.x;

  if (j_ptr < mode_1_ptr[mode_0_ptr[1]]) {
    int64_t i_ptr = -1;
    for (uint64_t idx = 0; idx < mode_0_ptr[1]; ++idx) {
      if (j_ptr >= mode_1_ptr[idx] && j_ptr < mode_1_ptr[idx + 1]) {
        i_ptr = idx;
        break;
      }
    }
    if (i_ptr < 0) return; // Out of bounds check

    if(i_ptr >= 0 && i_ptr < mode_0_ptr[1]){
      uint64_t i = mode_0_idx[i_ptr];

      for (uint64_t z = 0; z < n; ++z) {
        uint64_t k = z;

        if (k_buffer[j_ptr * n + k] > 0) {
          for (uint64_t r = 0; r < f1; ++r) {
            uint64_t index_A = i * f1 + r;

            for (uint64_t s = 0; s < f2; ++s) {
              uint64_t index_O = k * f1 * f2 + r * f2 + s;
              uint64_t index_buf = k * f2 + s;

              atomicAdd(&arr_O[index_O], buffer[j_ptr * n * f2 + index_buf] * arr_A[index_A]);
            }
          }
        }
      }
    }
  }
}

/*End of device function for GPU 4 loop Method*/
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/*Start of host function for GPU 4 loop Method*/
void gpu_factorize_n_fuse(
  uint64_t** mode_ptrs, uint64_t** mode_idxs, float* values,
  float* factor_matrices[], uint64_t factor_sizes[],
  float* arr_O, uint64_t arr_O_size,
  int ncm, uint64_t ranks[], int order,
  uint64_t size_mode_ptr[], uint64_t size_mode_idx[], uint64_t dimensions[])
{
  // Allocate device memory
  uint64_t *d_mode_0_ptr, *d_mode_0_idx, *d_mode_1_ptr, *d_mode_1_idx, *d_mode_2_ptr, *d_mode_2_idx;
  float *d_values, *d_arr_A, *d_arr_B, *d_arr_O;
  float* buffer_for_contraction_0_1;
  float* buffer_for_contraction_2;
  int* k_buffer_for_contraction_2;

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
  int n = dimensions[2];

  cudaMalloc(&d_mode_0_ptr, sizeof(uint64_t) * size_mode_ptr[0]);
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
  cudaMemcpy(d_mode_0_ptr, mode_ptrs[0], sizeof(uint64_t) * size_mode_ptr[0], cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_0_idx, mode_idxs[0], sizeof(uint64_t) * size_mode_idx[0], cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_1_ptr, mode_ptrs[1], sizeof(uint64_t) * size_mode_ptr[1], cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_1_idx, mode_idxs[1], sizeof(uint64_t) * size_mode_idx[1], cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_2_ptr, mode_ptrs[2], sizeof(uint64_t) * size_mode_ptr[2], cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_2_idx, mode_idxs[2], sizeof(uint64_t) * size_mode_idx[2], cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, sizeof(float) * total_values, cudaMemcpyHostToDevice);
  cudaMemcpy(d_arr_A, factor_matrices[idx_A], sizeof(float) * factor_sizes[idx_A], cudaMemcpyHostToDevice);
  cudaMemcpy(d_arr_B, factor_matrices[idx_B], sizeof(float) * factor_sizes[idx_B], cudaMemcpyHostToDevice);
  cudaMemset(d_arr_O, 0, sizeof(float) * arr_O_size);

  // Launch kernel
  int threadsPerBlock = 256;
  
  // parallelising 'j_ptr' :
  int blocksPerGrid = (size_mode_idx[1] + threadsPerBlock - 1) / threadsPerBlock;
  
  if(ncm == 0 || ncm == 1){
    // parallelising 'j_ptr' for contraction = 0 and contraction = 1 :
    cudaCheckError(cudaMalloc(&buffer_for_contraction_0_1, f2 * size_mode_idx[1] * sizeof(float)));
    cudaCheckError(cudaMemset(buffer_for_contraction_0_1, 0, f2 * size_mode_idx[1] * sizeof(float)));
    auto start = std::chrono::high_resolution_clock::now();
    // parallelising 'i_ptr' :

    contractionKernel_4<<<blocksPerGrid, threadsPerBlock>>>(
      d_mode_0_ptr, d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
      d_values, d_arr_A, d_arr_B, d_arr_O, 
      f1, f2, ncm, buffer_for_contraction_0_1);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    cout << "Method: GPU_FNF, Time: " << duration / 1000.0 << " ms" << endl;
    }
    else if(ncm == 2){
      // parallelising 'j_ptr' for contraction = 2 :
      cudaMalloc(&buffer_for_contraction_2, n * f2 * size_mode_idx[1] * sizeof(float));
      cudaMalloc(&k_buffer_for_contraction_2, n * size_mode_idx[1] * sizeof(int));
      
      // parallelising 'j_ptr' for contraction = 2 :
      cudaMemset(buffer_for_contraction_2, 0, n * f2 * size_mode_idx[1] * sizeof(float));
      cudaMemset(k_buffer_for_contraction_2, 0, n * size_mode_idx[1] * sizeof(int));

      auto start = std::chrono::high_resolution_clock::now();
      contractionKernel_for_second_contraction_part_1<<<blocksPerGrid, threadsPerBlock>>>(
        d_mode_0_ptr, d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
        d_values, d_arr_A, d_arr_B, d_arr_O, n, f1, f2, ncm, buffer_for_contraction_2, k_buffer_for_contraction_2);
      cudaDeviceSynchronize();
      contractionKernel_for_second_contraction_part_2<<<blocksPerGrid, threadsPerBlock>>>(
        d_mode_0_ptr, d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
        d_values, d_arr_A, d_arr_B, d_arr_O, n, f1, f2, ncm, buffer_for_contraction_2, k_buffer_for_contraction_2);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      cout << "Method: GPU_FNF, Time: " << duration / 1000.0 << " ms" << endl;
      cudaDeviceSynchronize();
    }


  // Copy results back to host
  cudaMemcpy(arr_O, d_arr_O, sizeof(float) * arr_O_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_mode_0_ptr);
  cudaFree(d_mode_0_idx);
  cudaFree(d_mode_1_ptr);
  cudaFree(d_mode_1_idx);
  cudaFree(d_mode_2_ptr);
  cudaFree(d_mode_2_idx);
  cudaFree(d_values);
  cudaFree(d_arr_A);
  cudaFree(d_arr_B);
  cudaFree(d_arr_O);

  if(ncm == 0 || ncm == 1) {
    cudaFree(buffer_for_contraction_0_1);
  } else if(ncm == 2) {
    cudaFree(buffer_for_contraction_2);
    cudaFree(k_buffer_for_contraction_2);
  }
  // cudaDeviceSynchronize();
}

/*End of host function for GPU 4 loop Method*/
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
      
      // Run this implementation (GPU Factorize N Fuse) first
      if (verbose) {
          cout << "Running GPU Factorize N Fuse implementation..." << endl;
      }
      auto start = std::chrono::high_resolution_clock::now();
      
      gpu_factorize_n_fuse(
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
            cout << "Running reference implementation (CPU factorize_n_fuse)..." << endl;
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
          cout << "GPU Factorize N Fuse execution time: " << duration / 1000.0 << " ms" << endl;
          if (verify) {
              cout << "Reference execution time: " << ref_duration / 1000.0 << " ms" << endl;
              cout << "Speedup over reference: " << (float)ref_duration / duration << "x" << endl;
              cout << "Result validation: " << (valid ? "PASSED" : "FAILED") << endl;
          }
      } else {
          if (verify) {
              cout << "Method: GPU_FnF, Time: " << duration / 1000.0 << " ms, Validation: " << (valid ? "PASSED" : "FAILED") << endl;
          } else {
              cout << "Method: GPU_FnF, Time: " << duration / 1000.0 << " ms" << endl;
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