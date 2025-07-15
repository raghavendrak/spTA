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
/*Start of device function for GPU 5 loop Method*/
__global__ void GPU_5loop_contraction_kernel_0(
  uint64_t* mode_0_ptr, uint64_t* mode_0_idx,
  uint64_t* mode_1_ptr, uint64_t* mode_1_idx,
  uint64_t* mode_2_ptr, uint64_t* mode_2_idx,
  Scalar* values, Scalar* arr_A, Scalar* arr_B,  
  Scalar* arr_O, uint64_t l, uint64_t m, uint64_t n, uint64_t f1, uint64_t f2, int contraction,
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
  Scalar value;
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
        if (contraction == 0) {
          index_A = j * f1 + r;
        } else if (contraction == 1) {
          index_A = i * f1 + r;
        } else if (contraction == 2) {
          index_A = i * f1 + r;
        }

        for (uint64_t s = 0; s < f2; ++s) {
          if (contraction == 0) {
            index_B = k * f2 + s;
            index_O = i * f1 * f2 + r * f2 + s;
          } else if (contraction == 1) {
            index_B = k * f2 + s;
            index_O = j * f1 * f2 + r * f2 + s;
          } else if (contraction == 2) {
            index_B = j * f2 + s;
            index_O = k * f1 * f2 + r * f2 + s;
          }

          atomicAdd(&arr_O[index_O], value * arr_A[index_A] * arr_B[index_B]);
        }
      }
    }
  }
}

/*End of device function for GPU 5 loop Method*/
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/*Start of host function for GPU 5 loop Method*/

void performContraction_gpu_1(
  uint64_t* mode_0_ptr, uint64_t* mode_0_idx,
  uint64_t* mode_1_ptr, uint64_t* mode_1_idx,
  uint64_t* mode_2_ptr, uint64_t* mode_2_idx,
  Scalar* values, Scalar* arr_A, Scalar* arr_B,
  Scalar* arr_O, uint64_t arr_A_size, uint64_t arr_B_size, uint64_t arr_O_size,
  int contraction, uint64_t dim_0, uint64_t dim_1, uint64_t dim_2,
  uint64_t r1, uint64_t r2, uint64_t total_values,
  uint64_t size_mode_0_ptr, uint64_t size_mode_1_ptr, uint64_t size_mode_2_ptr,
  uint64_t size_mode_0_idx, uint64_t size_mode_1_idx, uint64_t size_mode_2_idx)
{
  // Allocate device memory
  uint64_t *d_mode_0_ptr, *d_mode_0_idx, *d_mode_1_ptr, *d_mode_1_idx, *d_mode_2_ptr, *d_mode_2_idx;
  Scalar *d_values, *d_arr_A, *d_arr_B, *d_arr_O;

  cudaMalloc(&d_mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr);
  cudaMalloc(&d_mode_0_idx, sizeof(uint64_t) * size_mode_0_idx);
  cudaMalloc(&d_mode_1_ptr, sizeof(uint64_t) * size_mode_1_ptr);
  cudaMalloc(&d_mode_1_idx, sizeof(uint64_t) * size_mode_1_idx);
  cudaMalloc(&d_mode_2_ptr, sizeof(uint64_t) * size_mode_2_ptr);
  cudaMalloc(&d_mode_2_idx, sizeof(uint64_t) * size_mode_2_idx);
  cudaMalloc(&d_values, sizeof(Scalar) * total_values);
  cudaMalloc(&d_arr_A, sizeof(Scalar) * arr_A_size);
  cudaMalloc(&d_arr_B, sizeof(Scalar) * arr_B_size);
  cudaMalloc(&d_arr_O, sizeof(Scalar) * arr_O_size);

  // Copy data from host to device
  cudaMemcpy(d_mode_0_ptr, mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_0_idx, mode_0_idx, sizeof(uint64_t) * size_mode_0_idx, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_1_ptr, mode_1_ptr, sizeof(uint64_t) * size_mode_1_ptr, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_1_idx, mode_1_idx, sizeof(uint64_t) * size_mode_1_idx, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_2_ptr, mode_2_ptr, sizeof(uint64_t) * size_mode_2_ptr, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_2_idx, mode_2_idx, sizeof(uint64_t) * size_mode_2_idx, cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, sizeof(Scalar) * total_values, cudaMemcpyHostToDevice);
  cudaMemcpy(d_arr_A, arr_A, sizeof(Scalar) * arr_A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_arr_B, arr_B, sizeof(Scalar) * arr_B_size, cudaMemcpyHostToDevice);
  cudaMemset(d_arr_O, 0, sizeof(Scalar) * arr_O_size);

  // Kernel launch parameters
  int threadsPerBlock = 256;
  int blocksPerGrid = (size_mode_1_idx + threadsPerBlock - 1) / threadsPerBlock;

  // Launch appropriate kernel based on contraction type
  GPU_5loop_contraction_kernel_0<<<blocksPerGrid, threadsPerBlock>>>(
    d_mode_0_ptr, d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
    d_values, d_arr_A, d_arr_B, d_arr_O, dim_0, dim_1, dim_2, r1, r2, contraction,
    size_mode_0_ptr, size_mode_1_ptr, size_mode_2_ptr, size_mode_0_idx, size_mode_1_idx, size_mode_2_idx
  );

  // Check for launch errors
  cudaGetLastError();
  cudaDeviceSynchronize();

  // Copy results back to host
  cudaMemcpy(arr_O, d_arr_O, sizeof(Scalar) * arr_O_size, cudaMemcpyDeviceToHost);

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

/*End of host function for GPU 5 loop Method*/
////////////////////////////////////////////////////////////////////

// Include the reference implementation for validation
#define INCLUDED_AS_LIBRARY
#include "v2_cpu_4loop.cu"

int main(int argc, char* argv[]) {
    bool verbose = false;
    string csf_file;
    uint64_t rank1 = 30, rank2 = 30;
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
        
        // Convert CSF tensor to arrays
        uint64_t *mode_0_ptr, *mode_0_idx;
        uint64_t *mode_1_ptr, *mode_1_idx;
        uint64_t *mode_2_ptr, *mode_2_idx;
        Scalar *values;
        int order;
        
        size_t size_mode_0_ptr = tensor.ptrs[0].size();
        size_t size_mode_1_ptr = tensor.ptrs[1].size();
        size_t size_mode_2_ptr = tensor.ptrs[2].size();
        size_t size_mode_0_idx = tensor.idxs[0].size();
        size_t size_mode_1_idx = tensor.idxs[1].size();
        size_t size_mode_2_idx = tensor.idxs[2].size();
        size_t total_values = tensor.values.size();
        
        vector<uint64_t> dimensions(tensor.order);
        for(int i = 0; i < tensor.order; i++){
            dimensions[i] = tensor.dimensions[i];
        }

        getCSFArrays(tensor, &mode_0_ptr, &mode_0_idx, 
                    &mode_1_ptr, &mode_1_idx, 
                    &mode_2_ptr, &mode_2_idx, 
                    &values, &order);
        
        // Calculate matrix dimensions based on contraction mode
        uint64_t matrix_dim1 = getMatrixDim1(dimensions, ncm);
        uint64_t matrix_dim2 = getMatrixDim2(dimensions, ncm);
        uint64_t out_dim1 = getOutputDim1(dimensions, ncm);
        
        // Generate factor matrices
        Scalar *arr_A = nullptr, *arr_B = nullptr;
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
            cout << "Output dimensions: " << out_dim1 << " x " << out_dim2 << endl;
        }
        
        // Allocate output array
        Scalar* arr_O = allocate_aligned_array(arr_O_size);
        Scalar* ref_O = nullptr;
        
        if (verify) {
            // Only allocate reference array if verification is needed
            ref_O = allocate_aligned_array(arr_O_size);
        }
        
        // Run this implementation (GPU 5-loop) first
        if (verbose) {
            cout << "Running GPU 5-loop implementation..." << endl;
        }
        auto start = std::chrono::high_resolution_clock::now();
        
        performContraction_gpu_1(
            mode_0_ptr, mode_0_idx,
            mode_1_ptr, mode_1_idx,
            mode_2_ptr, mode_2_idx,
            values, arr_A, arr_B, arr_O,
            arr_A_size, arr_B_size, arr_O_size,
            ncm, dimensions[0], dimensions[1], dimensions[2], rank1, rank2,
            total_values,
            size_mode_0_ptr, size_mode_1_ptr, size_mode_2_ptr,
            size_mode_0_idx, size_mode_1_idx, size_mode_2_idx
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        bool valid = true;
        Scalar ref_duration = 0.0;
        
        if (verify) {
            // Now run reference implementation (CPU 4-loop) for validation
            if (verbose) {
                cout << "Running reference implementation (CPU 4-loop)..." << endl;
            }
            auto ref_start = std::chrono::high_resolution_clock::now();
            
            performContraction_cpu_2(
                mode_0_ptr, mode_0_idx,
                mode_1_ptr, mode_1_idx,
                mode_2_ptr, mode_2_idx,
                values, arr_A, arr_B, ref_O,
                arr_A_size, arr_B_size, arr_O_size, ncm,
                dimensions[0], dimensions[1], dimensions[2], rank1, rank2
            );
            
            auto ref_end = std::chrono::high_resolution_clock::now();
            ref_duration = std::chrono::duration_cast<std::chrono::microseconds>(ref_end - ref_start).count();
            
            // Validate results using compare_results from matrix_utils.h
            valid = compare_results(arr_O, ref_O, arr_O_size);
        }
        
        // Report results
        if (verbose) {
            cout << "GPU 5-loop execution time: " << duration / 1000.0 << " ms" << endl;
            if (verify) {
                cout << "Reference execution time: " << ref_duration / 1000.0 << " ms" << endl;
                cout << "Speedup over reference: " << (Scalar)ref_duration / duration << "x" << endl;
                cout << "Result validation: " << (valid ? "PASSED" : "FAILED") << endl;
            }
        } else {
            if (verify) {
                cout << "Method: GPU_5L, Time: " << duration / 1000.0 << " ms, Validation: " << (valid ? "PASSED" : "FAILED") << endl;
            } else {
                cout << "Method: GPU_5L, Time: " << duration / 1000.0 << " ms" << endl;
            }
        }
        
        // Clean up
        delete[] mode_0_ptr;
        delete[] mode_0_idx;
        delete[] mode_1_ptr;
        delete[] mode_1_idx;
        delete[] mode_2_ptr;
        delete[] mode_2_idx;
        delete[] values;
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
