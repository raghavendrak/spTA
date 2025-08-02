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
/*Start of device function for GPU All At Once Method*/
__global__ void GPU_AAO(
  uint64_t* mode_0_ptr, uint64_t* mode_0_idx,
  uint64_t* mode_1_ptr, uint64_t* mode_1_idx,
  uint64_t* mode_2_ptr, uint64_t* mode_2_idx,
  float* values, float* arr_A, float* arr_B,  
  float* arr_O, uint64_t f1, uint64_t f2, int ncm,
  uint64_t size_mode_0_ptr, uint64_t size_mode_1_ptr, uint64_t size_mode_2_ptr,
  uint64_t size_mode_0_idx, uint64_t size_mode_1_idx, uint64_t size_mode_2_idx) 
{
  // Compute thread index
  uint64_t j_ptr = blockIdx.x * blockDim.x + threadIdx.x;

  // Find the i_ptr associated with the current j_ptr
  int64_t i_ptr = -1;
  for (uint64_t p = 0; p < size_mode_1_ptr - 1; ++p) {
    if (mode_1_ptr[p] <= j_ptr && j_ptr < mode_1_ptr[p + 1]) {
      i_ptr = p;
      break;
    }
  }
  uint64_t i, j, k, index_A, index_B, index_O;
  float value;
  if ((i_ptr >= 0 && i_ptr < mode_0_ptr[1]) && 
      ( j_ptr < size_mode_1_idx) ) 
  {
    for (uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
    
      i = mode_0_idx[i_ptr] ;
      j = mode_1_idx[j_ptr] ;

      k = mode_2_idx[k_ptr] ;
      value = values[k_ptr];

      for (uint64_t r = 0; r < f1; ++r) {
        index_A = 0;
        if (ncm == 0) {
          index_A = j * f1 + r;
        } else if (ncm == 1) {
          index_A = i * f1 + r;
        } else if (ncm == 2) {
          index_A = i * f1 + r;
        }

        for (uint64_t s = 0; s < f2; ++s) {
          if (ncm == 0) {
            index_B = k * f2 + s;
            index_O = i * f1 * f2 + r * f2 + s;
          } else if (ncm == 1) {
            index_B = k * f2 + s;
            index_O = j * f1 * f2 + r * f2 + s;
          } else if (ncm == 2) {
            index_B = j * f2 + s;
            index_O = k * f1 * f2 + r * f2 + s;
          }

          atomicAdd(&arr_O[index_O], value * arr_A[index_A] * arr_B[index_B]);
        }
      }
    }
  }
}

/*End of device function for GPU All At Once Method*/
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/*Start of host function for GPU All At Once Method*/

void gpu_all_at_once(
  uint64_t** mode_ptrs, uint64_t** mode_idxs, float* values,
  float* factor_matrices[], uint64_t factor_sizes[],
  float* arr_O, uint64_t arr_O_size,
  int ncm, uint64_t ranks[], int order,
  uint64_t size_mode_ptr[], uint64_t size_mode_idx[])
{
  // Allocate device memory
  uint64_t *d_mode_0_ptr, *d_mode_0_idx, *d_mode_1_ptr, *d_mode_1_idx, *d_mode_2_ptr, *d_mode_2_idx;
  float *d_values, *d_arr_A, *d_arr_B, *d_arr_O;

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

  // Copy data from host to device
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

  // Kernel launch parameters
  int threadsPerBlock = 256;
  int blocksPerGrid = (size_mode_idx[1] + threadsPerBlock - 1) / threadsPerBlock;

  auto start = std::chrono::high_resolution_clock::now();
  // Launch appropriate kernel based on contraction type
  GPU_AAO<<<blocksPerGrid, threadsPerBlock>>>(
    d_mode_0_ptr, d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
    d_values, d_arr_A, d_arr_B, d_arr_O, ranks[0], ranks[1], ncm,
    size_mode_ptr[0], size_mode_ptr[1], size_mode_ptr[2], 
    size_mode_idx[0], size_mode_idx[1], size_mode_idx[2]
  );
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  cout << "Method: GPU_AAO, Time: " << duration / 1000.0 << " ms" << endl;

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
}

/*End of host function for GPU All At Once Method*/
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
        
        // Run this implementation (GPU All At Once) first
        if (verbose) {
            cout << "Running GPU All At Once implementation..." << endl;
        }
        auto start = std::chrono::high_resolution_clock::now();
        
        gpu_all_at_once(
            mode_ptrs.data(), mode_idxs.data(), values,
            factor_matrices.data(), factor_sizes.data(),
            arr_O, arr_O_size,
            ncm, ranks.data(), order,
            size_mode_ptr.data(), size_mode_idx.data()
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
            cout << "GPU All At Once execution time: " << duration / 1000.0 << " ms" << endl;
            if (verify) {
                cout << "Reference execution time: " << ref_duration / 1000.0 << " ms" << endl;
                cout << "Speedup over reference: " << (float)ref_duration / duration << "x" << endl;
                cout << "Result validation: " << (valid ? "PASSED" : "FAILED") << endl;
            }
        } else {
            if (verify) {
                cout << "Method: GPU_AAO, Time: " << duration / 1000.0 << " ms, Validation: " << (valid ? "PASSED" : "FAILED") << endl;
            } else {
                cout << "Method: GPU_AAO, Time: " << duration / 1000.0 << " ms" << endl;
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
