#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <chrono> 
#include <iomanip> 
#include <cuda_runtime.h>
#include "genten.h"      // Include the header for genten
#include "COO_to_CSF.h"

#include <cstdlib>  // for std::aligned_alloc, std::free, size_t
#include <cstring>  // for std::memset
#include <stdexcept> // for std::runtime_error

using namespace std;
using namespace std::chrono;


//////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////
// Function to decrement each element by 1 to convert tensor from 1 based indexing to 0 based indexing
void decrementArray(uint64_t* arr, uint64_t size) {
  for (uint64_t i = 0; i < size; ++i) {
      arr[i] -= 1;
  }
}
//////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////
/* Start of Function for reading a matrix from .txt file*/

// Function to read a matrix from a file 
void readMatrix(const string& filename, uint64_t& rows, uint64_t& cols, double*& arr) {
  ifstream file(filename);
  if (!file.is_open()) {
    throw runtime_error("Unable to open matrix file: " + filename);
  }

  // Read the entire file into a vector
  arr = new double[rows * cols];
  double value;
  
  uint64_t count = 0;
  while (file >> value) {
    if (count < rows * cols) {
      arr[count++] = value;
    } else {
      throw runtime_error("More values in the file than expected.");
    }
  }
  
  // Close the file
  file.close();

  if (count % cols != 0) {
    throw runtime_error("Mismatch between total number of elements and specified column count.");
  }
}

/* End of Function for reading a matrix from .txt file*/
/////////////////////////////////////////////////////


//////////////////////////////////////////////////////////
/* Start of Function for writing a matrix from .txt file*/

void writeMatrixToFile(const std::string& filename, uint64_t rows, uint64_t cols, unsigned int seed) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filename << std::endl;
    return;
  }

  std::mt19937 gen(seed); // Mersenne Twister RNG seeded with 'seed'
  std::uniform_real_distribution<double> dist(0.0, 1.0); // Uniform distribution in [0, 1)

  for (uint64_t i = 0; i < rows; ++i) {
    for (uint64_t j = 0; j < cols; ++j) {
      file << dist(gen) << " ";
    }
    file << "\n";
  }

  file.close();
}
/* End of Function for writing a matrix from .txt file*/
////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
/* Start of Function for comparing the results of various contractions*/
bool compare_matrices(double*& C1, double*& C2, int rows, int cols, double tolerance = 1e-6)
{
  for (int i = 0; i < rows * cols; ++i) {
    if (std::fabs(C1[i] - C2[i]) > tolerance) {
      std::cout << " NOT matching at i : " << i << endl;
      return false;
    }
  }
  return true;
}

/*End of Function for comparing the results of various contractions*/
/////////////////////////////////////////////////////////////////////

__device__ double atomicAdd_double(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

/////////////////////////////////////////////////////////////////////
/*Start of aligned allocation*/

double* allocate_aligned_array(size_t num_elements) {
    constexpr size_t alignment = 32;           // 32 bytes = 256 bits
    constexpr size_t element_size = sizeof(double); // 8 bytes per double

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
    double* arr = static_cast<double*>(ptr);
    for (size_t i = 0; i < total_elements; ++i) {
      arr[i] = 0.0;
    }

    return static_cast<double*>(ptr);
}
/*End of aligned allocation*/
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/*Start of CPU 5 loop Method*/
void performContraction_cpu_1(uint64_t*& mode_0_ptr, uint64_t*& mode_0_idx,
                        uint64_t*& mode_1_ptr, uint64_t*& mode_1_idx,
                        uint64_t*& mode_2_ptr, uint64_t*& mode_2_idx,
                        double*& values, double*& arr_A, double*& arr_B,  
                        double*& arr_O, uint64_t& arr_A_size, uint64_t& arr_B_size, uint64_t& arr_O_size, int& contraction, 
                        uint64_t& l, uint64_t& m, uint64_t& n, uint64_t& f1, uint64_t& f2) 
{
  uint64_t i, j, k, index_A, index_B, index_O;
  double value;
  if(contraction == 0){
    // Traverse through CSF tensor pointer and indices arrays for all modes
    for (uint64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
      i = mode_0_idx[i_ptr];                         // Index in the mode 'i'

      for (uint64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
        j = mode_1_idx[j_ptr];                     // Index for 'j' mode in CSF

        for (uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
          k = mode_2_idx[k_ptr];                 // Index for 'k' mode in CSF

          value = values[k_ptr];                  // CSF value for the above i, j, k

          // Iterate over the matrix dimensions 
          for (uint64_t r = 0; r < f1; ++r) {
              
            index_A = j * f1 + r;
            for (uint64_t s = 0; s < f2; ++s) {

              // Compute linearized indices for matrices A, B based on the contraction string
              index_B = k * f2 + s;

              // For mode-1 linearized output 
              index_O = i * f1 * f2 + r * f2 + s;

              // Perform contraction
              arr_O[index_O] += value * arr_A[index_A] * arr_B[index_B];
                
            }
          }
        }
      }
    }
  }
  else if(contraction == 1){
    // Traverse through CSF tensor pointer and indices arrays for all modes
    for (uint64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
      i = mode_0_idx[i_ptr];                         // Index in the mode 'i'

      for (uint64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
        j = mode_1_idx[j_ptr];                     // Index for 'j' mode in CSF

        for (uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
          k = mode_2_idx[k_ptr];                 // Index for 'k' mode in CSF

          value = values[k_ptr];                  // CSF value for the above i, j, k

          // Iterate over the matrix dimensions 
          for (uint64_t r = 0; r < f1; ++r) {
              
            index_A = i * f1 + r;
            for (uint64_t s = 0; s < f2; ++s) {

              // Compute linearized indices for matrices A, B based on the contraction string
              index_B = k * f2 + s;

              // For mode-1 linearized output 
              index_O = j * f1 * f2 + r * f2 + s;

              // Perform contraction
              arr_O[index_O] += value * arr_A[index_A] * arr_B[index_B];            
            }
          }
        }
      }
    }
  }
  else if(contraction == 2){
    // Traverse through CSF tensor pointer and indices arrays for all modes
    for (uint64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
      i = mode_0_idx[i_ptr] ;                         // Index in the mode 'i'

      for (uint64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
        j = mode_1_idx[j_ptr] ;                     // Index for 'j' mode in CSF

        for (uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
          k = mode_2_idx[k_ptr] ;                 // Index for 'k' mode in CSF

          value = values[k_ptr];                  // CSF value for the above i, j, k

          // Iterate over the matrix dimensions 
          for (uint64_t r = 0; r < f1; ++r) {
              
            index_A = i * f1 + r;
            for (uint64_t s = 0; s < f2; ++s) {

              // Compute linearized indices for matrices A, B based on the contraction string
              index_B = j * f2 + s;

              // For mode-1 linearized output
              index_O = k * f1 * f2 + r * f2 + s;

              // Perform contraction
              arr_O[index_O] += value * arr_A[index_A] * arr_B[index_B];            
            }
          }
        }
      }
    }
  }
}
/*End of CPU 5 loop Method*/
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/*Start of CPU 4 loop Method*/

// Function to perform contraction based on the inputs using 4 for loops
void performContraction_cpu_2(uint64_t*& mode_0_ptr,uint64_t*& mode_0_idx,
                        uint64_t*& mode_1_ptr, uint64_t*& mode_1_idx,
                        uint64_t*& mode_2_ptr, uint64_t*& mode_2_idx,
                        double*& values, double*& arr_A, double*& arr_B,  
                        double*& arr_O, uint64_t& arr_A_size, uint64_t& arr_B_size, uint64_t& arr_O_size, int& contraction, 
                        uint64_t& l, uint64_t& m, uint64_t& n, uint64_t& f1, uint64_t& f2) 
{
  uint64_t i, j, k, index_A, index_B, index_O;
  double value;
                            
  if(contraction == 0){
    double* buffer = allocate_aligned_array(f2);    // buffer for mode-s

    // Traverse through CSF tensor pointer and indices arrays for all modes
    for (uint64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
      i = mode_0_idx[i_ptr] ;                         // Index in the mode 'i'
      
      for (uint64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
        j = mode_1_idx[j_ptr] ;                     // Index for 'j' mode in CSF
        
        memset(buffer, 0, f2 * sizeof(double));             // Set the entire memory block to 0
        
        for (uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
          k = mode_2_idx[k_ptr] ;                 // Index for 'k' mode in CSF

          value = values[k_ptr];                  // CSF value for the above i, j, k

          for (uint64_t s = 0; s < f2; ++s) {

            // Compute linearized indices for matrices B based on the contraction string
            index_B = k * f2 + s;

            buffer[s] += value * arr_B[index_B];                        
          }
        }

        for (uint64_t r = 0; r < f1; ++r) {
            
          // Compute linearized indices for matrices B based on the contraction string
          index_A = j * f1 + r;
          for (uint64_t s = 0; s < f2; ++s) {

            // For mode-1 linearized output
            index_O = i * f1 * f2 + r * f2 + s;

            // Perform contraction
            arr_O[index_O] += buffer[s] * arr_A[index_A];              
          }
        }
      }
    }
    std::free(buffer);
  }
  else if(contraction == 1){
    double* buffer = allocate_aligned_array(f2);    // buffer for mode-s
    
    // Traverse through CSF tensor pointer and indices arrays for all modes
    for (uint64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
      i = mode_0_idx[i_ptr] ;                         // Index in the mode 'i'

      for (uint64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
        j = mode_1_idx[j_ptr] ;                     // Index for 'j' mode in CSF

        memset(buffer, 0, f2 * sizeof(double));             // Set the entire memory block to 0

        for (uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
          k = mode_2_idx[k_ptr] ;                 // Index for 'k' mode in CSF

          value = values[k_ptr];                  // CSF value for the above i, j, k

          for (uint64_t s = 0; s < f2; ++s) {

            // Compute linearized indices for matrices B based on the contraction string
            index_B = k * f2 + s;

            // Perform contraction
            buffer[s] += value * arr_B[index_B];            
          }
        }

        for (uint64_t r = 0; r < f1; ++r) {
          // Compute linearized indices for matrices A, B based on the contraction string
          index_A = i * f1 + r;
          for (uint64_t s = 0; s < f2; ++s) {

            // For mode-1 linearized output 
            index_O = j * f1 * f2 + r * f2 + s;

            // Perform contraction
            arr_O[index_O] += buffer[s] * arr_A[index_A];              
          }
        }
      }
    }
    std::free(buffer);
  }
  else if(contraction == 2){
    double* buffer = allocate_aligned_array(n*f2);    // buffer for mode-k and mode-s
    bool* k_buffer = new bool[n];  // buffer for k-indices
    uint64_t index_buf = 0;

    // Traverse through CSF tensor pointer and indices arrays for all modes
    for (uint64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
      i = mode_0_idx[i_ptr] ;                          // Index in the mode 'i'

      memset(buffer, 0, n * f2 * sizeof(double));             // Set the entire memory block to 0
      memset(k_buffer, 0, n * sizeof(bool)); //initialize to false
      for (uint64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
        j = mode_1_idx[j_ptr] ;                      // Index for 'j' mode in CSF

        for (uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
          k = mode_2_idx[k_ptr] ;                  // Index for 'k' mode in CSF
          k_buffer[k] = true;

          value = values[k_ptr];                   // CSF value for the above i, j, k

          for (uint64_t s = 0; s < f2; ++s) {

            // Compute linearized indices for matrices B based on the contraction string
            index_B = j * f2 + s;

            index_buf = k * f2 + s; 

            buffer[index_buf] += value * arr_B[index_B];
          }
        }
      }

      for (uint64_t z = 0; z < n ; ++z) {
        if(k_buffer[z]){
          k = z;
          for (uint64_t r = 0; r < f1; ++r) {
  
            // Compute linearized indices for matrices A based on the contraction string
            index_A = i * f1 + r;
            for (uint64_t s = 0; s < f2; ++s) {
        
              // For mode-1 linearized output 
              index_O = k * f1 * f2 + r * f2 + s;
  
              index_buf = k * f2 + s; 
  
              arr_O[index_O] += buffer[index_buf] * arr_A[index_A] ;        
            }
          }
        }
      }
    }
    std::free(buffer);
    delete [] k_buffer;
  } 
}

/*End of CPU 4 loop Method*/
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/*Start of device function for GPU 5 loop Method*/
__global__ void GPU_5loop_contraction_kernel_0(
  uint64_t* mode_0_ptr, uint64_t* mode_0_idx,
  uint64_t* mode_1_ptr, uint64_t* mode_1_idx,
  uint64_t* mode_2_ptr, uint64_t* mode_2_idx,
  double* values, double* arr_A, double* arr_B,  
  double* arr_O, uint64_t l, uint64_t m, uint64_t n, uint64_t f1, uint64_t f2, int contraction,
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
  double value;
  if ((i_ptr >= 0 && i_ptr < mode_0_ptr[1]) && 
      (j_ptr >= 0 && j_ptr < size_mode_1_idx) ) 
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

          atomicAdd_double(&arr_O[index_O], value * arr_A[index_A] * arr_B[index_B]);
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
  double* values, double* arr_A, double* arr_B,
  double* arr_O, uint64_t arr_A_size, uint64_t arr_B_size, uint64_t arr_O_size,
  int contraction, uint64_t dim_0, uint64_t dim_1, uint64_t dim_2,
  uint64_t r1, uint64_t r2, uint64_t total_values,
  uint64_t size_mode_0_ptr, uint64_t size_mode_1_ptr, uint64_t size_mode_2_ptr,
  uint64_t size_mode_0_idx, uint64_t size_mode_1_idx, uint64_t size_mode_2_idx)
{
  // Allocate device memory
  uint64_t *d_mode_0_ptr, *d_mode_0_idx, *d_mode_1_ptr, *d_mode_1_idx, *d_mode_2_ptr, *d_mode_2_idx;
  double *d_values, *d_arr_A, *d_arr_B, *d_arr_O;

  cudaMalloc(&d_mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr);
  cudaMalloc(&d_mode_0_idx, sizeof(uint64_t) * size_mode_0_idx);
  cudaMalloc(&d_mode_1_ptr, sizeof(uint64_t) * size_mode_1_ptr);
  cudaMalloc(&d_mode_1_idx, sizeof(uint64_t) * size_mode_1_idx);
  cudaMalloc(&d_mode_2_ptr, sizeof(uint64_t) * size_mode_2_ptr);
  cudaMalloc(&d_mode_2_idx, sizeof(uint64_t) * size_mode_2_idx);
  cudaMalloc(&d_values, sizeof(double) * total_values);
  cudaMalloc(&d_arr_A, sizeof(double) * arr_A_size);
  cudaMalloc(&d_arr_B, sizeof(double) * arr_B_size);
  cudaMalloc(&d_arr_O, sizeof(double) * arr_O_size);

  // Copy data from host to device
  cudaMemcpy(d_mode_0_ptr, mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_0_idx, mode_0_idx, sizeof(uint64_t) * size_mode_0_idx, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_1_ptr, mode_1_ptr, sizeof(uint64_t) * size_mode_1_ptr, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_1_idx, mode_1_idx, sizeof(uint64_t) * size_mode_1_idx, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_2_ptr, mode_2_ptr, sizeof(uint64_t) * size_mode_2_ptr, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_2_idx, mode_2_idx, sizeof(uint64_t) * size_mode_2_idx, cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, sizeof(double) * total_values, cudaMemcpyHostToDevice);
  cudaMemcpy(d_arr_A, arr_A, sizeof(double) * arr_A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_arr_B, arr_B, sizeof(double) * arr_B_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_arr_O, arr_O, sizeof(double) * arr_O_size, cudaMemcpyHostToDevice);

  // Kernel launch parameters
  int threadsPerBlock = 256;
  int blocksPerGrid = (size_mode_2_idx + threadsPerBlock - 1) / threadsPerBlock;

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
  cudaMemcpy(arr_O, d_arr_O, sizeof(double) * arr_O_size, cudaMemcpyDeviceToHost);

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
  
/*
{
  // Allocate device memory
  uint64_t *d_mode_0_ptr, *d_mode_0_idx, *d_mode_1_ptr, *d_mode_1_idx, *d_mode_2_ptr, *d_mode_2_idx;
  double *d_values, *d_arr_A, *d_arr_B, *d_arr_O;

  cudaCheckError(cudaMalloc(&d_mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr));
  cudaCheckError(cudaMalloc(&d_mode_0_idx, sizeof(uint64_t) * size_mode_0_idx));
  cudaCheckError(cudaMalloc(&d_mode_1_ptr, sizeof(uint64_t) * size_mode_1_ptr));
  cudaCheckError(cudaMalloc(&d_mode_1_idx, sizeof(uint64_t) * size_mode_1_idx));
  cudaCheckError(cudaMalloc(&d_mode_2_ptr, sizeof(uint64_t) * size_mode_2_ptr));
  cudaCheckError(cudaMalloc(&d_mode_2_idx, sizeof(uint64_t) * size_mode_2_idx));
  cudaCheckError(cudaMalloc(&d_values, sizeof(double) * total_values));
  cudaCheckError(cudaMalloc(&d_arr_A, sizeof(double) * arr_A_size));
  cudaCheckError(cudaMalloc(&d_arr_B, sizeof(double) * arr_B_size));
  cudaCheckError(cudaMalloc(&d_arr_O, sizeof(double) * arr_O_size));

  // Copy data from host to device
  cudaCheckError(cudaMemcpy(d_mode_0_ptr, mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_mode_0_idx, mode_0_idx, sizeof(uint64_t) * size_mode_0_idx, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_mode_1_ptr, mode_1_ptr, sizeof(uint64_t) * size_mode_1_ptr, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_mode_1_idx, mode_1_idx, sizeof(uint64_t) * size_mode_1_idx, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_mode_2_ptr, mode_2_ptr, sizeof(uint64_t) * size_mode_2_ptr, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_mode_2_idx, mode_2_idx, sizeof(uint64_t) * size_mode_2_idx, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_values, values, sizeof(double) * total_values, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_arr_A, arr_A, sizeof(double) * arr_A_size, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_arr_B, arr_B, sizeof(double) * arr_B_size, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_arr_O, arr_O, sizeof(double) * arr_O_size, cudaMemcpyHostToDevice));

  // Kernel launch parameters
  int threadsPerBlock = 256;
  int blocksPerGrid = (size_mode_1_idx + threadsPerBlock - 1) / threadsPerBlock;

  // Launch appropriate kernel based on contraction type
  GPU_5loop_contraction_kernel_0<<<blocksPerGrid, threadsPerBlock>>>(
      d_mode_0_ptr, d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
      d_values, d_arr_A, d_arr_B, d_arr_O, dim_0, dim_1, dim_2, r1, r2, contraction);

  // Check for launch errors
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());

  // Copy results back to host
  cudaCheckError(cudaMemcpy(arr_O, d_arr_O, sizeof(double) * arr_O_size, cudaMemcpyDeviceToHost));

  // Free device memory
  cudaCheckError(cudaFree(d_mode_0_ptr));
  cudaCheckError(cudaFree(d_mode_0_idx));
  cudaCheckError(cudaFree(d_mode_1_ptr));
  cudaCheckError(cudaFree(d_mode_1_idx));
  cudaCheckError(cudaFree(d_mode_2_ptr));
  cudaCheckError(cudaFree(d_mode_2_idx));
  cudaCheckError(cudaFree(d_values));
  cudaCheckError(cudaFree(d_arr_A));
  cudaCheckError(cudaFree(d_arr_B));
  cudaCheckError(cudaFree(d_arr_O));
}
  */


/*End of host function for GPU 5 loop Method*/
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/*Start of device function for GPU 4 loop Method*/
__global__ void contractionKernel_4(
  uint64_t* mode_0_ptr, uint64_t* mode_0_idx,
  uint64_t* mode_1_ptr, uint64_t* mode_1_idx,
  uint64_t* mode_2_ptr, uint64_t* mode_2_idx,
  double* values, double* arr_A, double* arr_B,  
  double* arr_O, uint64_t l, uint64_t m, uint64_t n, uint64_t f1, uint64_t f2, int contraction, double* buffer)
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
        //double* buffer = (double*)malloc(f2 * sizeof(double));
        if (buffer == nullptr) {// Handle allocation failure
          printf("Memory allocation failure \n");
          return;
        } 

        // Reset buffer
        // memset(buffer, 0, f2 * sizeof(double));

        // Process the k_ptr range associated with j_ptr
        for (uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
          uint64_t k = mode_2_idx[k_ptr];
          double value = values[k_ptr];

          for (uint64_t s = 0; s < f2; ++s) {
            uint64_t index_B = k * f2 + s;
            // atomicAdd_double(&buffer[s], value * arr_B[index_B]);
            atomicAdd_double(&buffer[j_ptr * f2 + s], value * arr_B[index_B]);
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
            // atomicAdd_double(&arr_O[index_O], buffer[s] * arr_A[index_A]);
            atomicAdd_double(&arr_O[index_O], buffer[j_ptr * f2 + s] * arr_A[index_A]);
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
  double* values, double* arr_A, double* arr_B,  
  double* arr_O, uint64_t l, uint64_t m, uint64_t n, uint64_t f1, uint64_t f2, int contraction, double* buffer, int* k_buffer)
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
        double value = values[k_ptr];

        for (uint64_t s = 0; s < f2; ++s) {
          uint64_t index_B = j * f2 + s;
          uint64_t index_buf = k * f2 + s;

          // if (index_B >= n * f2 || j_ptr * (n * f2) + index_buf >= n * f2 * mode_1_ptr[mode_0_ptr[1]]) {
          //     printf("Out of bound access! \n");
          // }
          atomicAdd_double(&buffer[j_ptr * (n * f2) + index_buf], value * arr_B[index_B]);
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
  double* values, double* arr_A, double* arr_B,  
  double* arr_O, uint64_t l, uint64_t m, uint64_t n, uint64_t f1, uint64_t f2, int contraction, double* buffer, int* k_buffer)
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

              atomicAdd_double(&arr_O[index_O], buffer[j_ptr * n * f2 + index_buf] * arr_A[index_A]);
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
void performContraction_gpu_2(
  uint64_t* mode_0_ptr, uint64_t* mode_0_idx,
  uint64_t* mode_1_ptr, uint64_t* mode_1_idx,
  uint64_t* mode_2_ptr, uint64_t* mode_2_idx,
  double* values, double* arr_A, double* arr_B,  
  double* arr_O, uint64_t arr_A_size, uint64_t arr_B_size, uint64_t arr_O_size, int contraction, 
  uint64_t l, uint64_t m, uint64_t n, uint64_t f1, uint64_t f2, uint64_t total_values,
  int size_mode_0_ptr, int size_mode_1_ptr, int size_mode_2_ptr,
  int size_mode_0_idx, int size_mode_1_idx, int size_mode_2_idx)
{
  // Allocate device memory
  uint64_t *d_mode_0_ptr, *d_mode_0_idx, *d_mode_1_ptr, *d_mode_1_idx, *d_mode_2_ptr, *d_mode_2_idx;
  double *d_values, *d_arr_A, *d_arr_B, *d_arr_O;
  double* buffer_for_contraction_0_1;
  double* buffer_for_contraction_2;
  int* k_buffer_for_contraction_2;

  cudaMalloc(&d_mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr);
  cudaMalloc(&d_mode_0_idx, sizeof(uint64_t) * size_mode_0_idx);
  cudaMalloc(&d_mode_1_ptr, sizeof(uint64_t) * size_mode_1_ptr);
  cudaMalloc(&d_mode_1_idx, sizeof(uint64_t) * size_mode_1_idx);
  cudaMalloc(&d_mode_2_ptr, sizeof(uint64_t) * size_mode_2_ptr);
  cudaMalloc(&d_mode_2_idx, sizeof(uint64_t) * size_mode_2_idx);
  cudaMalloc(&d_values, sizeof(double) * total_values);
  cudaMalloc(&d_arr_A, sizeof(double) * arr_A_size);
  cudaMalloc(&d_arr_B, sizeof(double) * arr_B_size);
  cudaMalloc(&d_arr_O, sizeof(double) * arr_O_size);


  // parallelising 'j_ptr' for contraction = 0 and contraction = 1 :
  cudaMalloc(&buffer_for_contraction_0_1, f2 * size_mode_1_idx * sizeof(double));

  // parallelising 'j_ptr' for contraction = 2 :
  cudaMalloc(&buffer_for_contraction_2, n * f2 * size_mode_1_idx * sizeof(double));
  cudaMalloc(&k_buffer_for_contraction_2, n * size_mode_1_idx * sizeof(int));

  // Copy data to device
  cudaMemcpy(d_mode_0_ptr, mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_0_idx, mode_0_idx, sizeof(uint64_t) * size_mode_0_idx, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_1_ptr, mode_1_ptr, sizeof(uint64_t) * size_mode_1_ptr, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_1_idx, mode_1_idx, sizeof(uint64_t) * size_mode_1_idx, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_2_ptr, mode_2_ptr, sizeof(uint64_t) * size_mode_2_ptr, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mode_2_idx, mode_2_idx, sizeof(uint64_t) * size_mode_2_idx, cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, sizeof(double) * total_values, cudaMemcpyHostToDevice);
  cudaMemcpy(d_arr_A, arr_A, sizeof(double) * arr_A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_arr_B, arr_B, sizeof(double) * arr_B_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_arr_O, arr_O, sizeof(double) * arr_O_size, cudaMemcpyHostToDevice);


  // parallelising 'j_ptr' for contraction = 0 and contraction = 1 :
  cudaMemset(buffer_for_contraction_0_1, 0, f2 * size_mode_1_idx * sizeof(double));


  // parallelising 'j_ptr' for contraction = 2 :
  cudaMemset(buffer_for_contraction_2, 0, n * f2 * size_mode_1_idx * sizeof(double));
  cudaMemset(k_buffer_for_contraction_2, 0, n * size_mode_1_idx * sizeof(int));

  // Launch kernel
  int threadsPerBlock = 256;

  // parallelising 'j_ptr' :
  int blocksPerGrid = (size_mode_1_idx + threadsPerBlock - 1) / threadsPerBlock;

  if(contraction == 0 || contraction == 1){

    // parallelising 'i_ptr' :
    contractionKernel_4<<<blocksPerGrid, threadsPerBlock>>>(
        d_mode_0_ptr, d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
        d_values, d_arr_A, d_arr_B, d_arr_O, l, m, n, f1, f2, contraction, buffer_for_contraction_0_1);
  }
  else if(contraction == 2){
    contractionKernel_for_second_contraction_part_1<<<blocksPerGrid, threadsPerBlock>>>(
        d_mode_0_ptr, d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
        d_values, d_arr_A, d_arr_B, d_arr_O, l, m, n, f1, f2, contraction, buffer_for_contraction_2, k_buffer_for_contraction_2);
  }

  cudaDeviceSynchronize();

  if(contraction == 2){
    contractionKernel_for_second_contraction_part_2<<<blocksPerGrid, threadsPerBlock>>>(
        d_mode_0_ptr, d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
        d_values, d_arr_A, d_arr_B, d_arr_O, l, m, n, f1, f2, contraction, buffer_for_contraction_2, k_buffer_for_contraction_2);
  }


  // Copy results back to host
  cudaMemcpy(arr_O, d_arr_O, sizeof(double) * arr_O_size, cudaMemcpyDeviceToHost);

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

  cudaFree(buffer_for_contraction_0_1);
  cudaFree(buffer_for_contraction_2);
  cudaFree(k_buffer_for_contraction_2);
}

/*End of host function for GPU 4 loop Method*/
/////////////////////////////////////////////////////////////////////
int main(int argc, char** argv){
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << "order <dim_0> <dim_1> <dim_2> <rank_1> <rank_2> <ncm>" << endl;
    std::cerr << "order = Order of the tensor" << endl; //It is required for the genten to generate the tensor with specified order
    std::cerr << "dim_0 = Dimension 0, dim_1 = Dimension 1, dim_2 = Dimension 2, " << endl;
    std::cerr << "rank_1 = rank of 1st factor matrix (default = 30) \n rank_2 = rank of 2nd factor matrix (default = 30)" << endl;
    std::cerr << "ncm = Non Contracting Mode (default = 0)" << endl;
    std::cerr << "" << endl;
    std::cerr << "" << endl;
    return 1;
  }

  int default_rank = 30;
  int order = atoi(argv[1]);
  uint64_t dim_0 = atoi(argv[2]);
  uint64_t dim_1 = atoi(argv[3]);
  uint64_t dim_2 = atoi(argv[4]);

  uint64_t r1 = (argc > 4) ? atoi(argv[5]) : default_rank;
  uint64_t r2 = (argc > 5) ? atoi(argv[6]) : default_rank;
  int ncm = (argc > 6) ? atoi(argv[7]) : 0;

  if(ncm < 0 || ncm > 2){
    std::cerr << "Error: Contraction value must be 0, 1, or 2.\n";
    return 1;
  }
  std::cout <<"The Tensor is of dimension: " << dim_0 << "x" << dim_1 << "x" << dim_2 << endl;
  std::cout << "The column dimensions of output factor matrices  (r1 and r2) will be : " << r1 << r2 << endl;
  if(ncm == 0){
    cout << "Your Contraction Choice : ijk,jr,ks→irs" << endl; 
  }
  else if(ncm == 1){
    cout << "Your Contraction Choice : ijk,ir,ks→rjs" << endl;
  }
  else if(ncm == 2){
    cout << "Your Contraction Choice : ijk,ir,js→rsk" << endl;
  }
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  uint64_t rows_A, cols_A = r1, rows_B, cols_B = r2;

  if (ncm == 0) {
    rows_A = dim_1;
    rows_B = dim_2;
  } else if (ncm == 1) {
    rows_A = dim_0;
    rows_B = dim_2;
  } else if (ncm == 2) {
    rows_A = dim_0;
    rows_B = dim_1;
  }
  unsigned int A_seed = 1, B_seed = 2;
  // Write matrices to files
  writeMatrixToFile("input_matrix_A.txt", rows_A, cols_A, A_seed);
  writeMatrixToFile("input_matrix_B.txt", rows_B, cols_B, B_seed);

  std::cout << "Matrices written to input_matrix_A.txt and input_matrix_B.txt with dimensions:\n";
  std::cout << "Matrix A: " << rows_A << " x " << cols_A << "\n";
  std::cout << "Matrix B: " << rows_B << " x " << cols_B << "\n";

  
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  uint64_t* my_tensor_indices = nullptr;
  double* my_tensor_values = nullptr;
  uint64_t total_indices = 0;
  uint64_t total_values = 0;
  
  generate_tensor(argc, argv, &my_tensor_indices, &my_tensor_values, &total_indices, &total_values);
  
  std::cout << "\nOrder of the Tensor : " << order << endl;
  std::cout << "Dimension - 0 : " << dim_0 << endl;
  std::cout << "Dimension - 1 : " << dim_1 << endl;
  std::cout << "Dimension - 2 : " << dim_2 << endl;
  std::cout << "Total size of my_tensor_indices : " << total_indices<< endl;
  std::cout << "Total size of my_tensor_values : " << total_values << endl;

  cooToCSF(my_tensor_indices, my_tensor_values, order, total_indices, total_values);

  // Input tensor dimensions (l * m * n)
  uint64_t l, m, n;

  l = dim_0;
  m = dim_1;
  n = dim_2;

  uint64_t* mode_0_ptr = nullptr;
  uint64_t* mode_0_idx = nullptr;
  uint64_t* mode_1_ptr = nullptr;
  uint64_t* mode_1_idx = nullptr;
  uint64_t* mode_2_ptr = nullptr;
  uint64_t* mode_2_idx = nullptr;
  double* values = my_tensor_values;

  int size_mode_0_ptr = 0, size_mode_0_idx = 0;
  int size_mode_1_ptr = 0, size_mode_1_idx = 0;
  int size_mode_2_ptr = 0, size_mode_2_idx = 0;

  get_mode_0_ptr(&mode_0_ptr, &size_mode_0_ptr);
  get_mode_0_idx(&mode_0_idx, &size_mode_0_idx);
  get_mode_1_ptr(&mode_1_ptr, &size_mode_1_ptr);
  get_mode_1_idx(&mode_1_idx, &size_mode_1_idx);
  get_mode_2_ptr(&mode_2_ptr, &size_mode_2_ptr);
  get_mode_2_idx(&mode_2_idx, &size_mode_2_idx);

  decrementArray(mode_0_idx, size_mode_0_idx);
  decrementArray(mode_1_idx, size_mode_1_idx);
  decrementArray(mode_2_idx, size_mode_2_idx);

  cout << "Size of Mode 0 Pointer : " << size_mode_0_ptr << endl; 
  cout << "Size of Mode 1 Pointer : " << size_mode_1_ptr << endl; 
  cout << "Size of Mode 2 Pointer : " << size_mode_2_ptr << endl; 
  cout << "Size of Mode 0 Indices : " << size_mode_0_idx << endl; 
  cout << "Size of Mode 1 Indices : " << size_mode_1_idx << endl; 
  cout << "Size of Mode 2 Indices : " << size_mode_2_idx << endl; 

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  double* arr_A = nullptr;
  double* arr_B = nullptr;

  uint64_t arr_A_rows = 0;
  uint64_t arr_B_rows = 0;


  if (ncm == 0) {
    arr_A = allocate_aligned_array(dim_1 * r1);
    arr_B = allocate_aligned_array(dim_2 * r2);
    arr_A_rows = dim_1;
    arr_B_rows = dim_2;
  } else if (ncm == 1) {
    arr_A = allocate_aligned_array(dim_0 * r1);
    arr_B = allocate_aligned_array(dim_2 * r2);
    arr_A_rows = dim_0;
    arr_B_rows = dim_2;
  } else if (ncm == 2) {
    arr_A = allocate_aligned_array(dim_0 * r1);
    arr_B = allocate_aligned_array(dim_1 * r2);
    arr_A_rows = dim_0;
    arr_B_rows = dim_1;
  }

  readMatrix("input_matrix_A.txt", arr_A_rows, r1, arr_A);
  readMatrix("input_matrix_B.txt", arr_B_rows, r2, arr_B);

  uint64_t arr_A_size = arr_A_rows * r1;
  uint64_t arr_B_size = arr_B_rows * r2;

  uint64_t output_sizes[3];
  output_sizes[0] = dim_0 * r1 * r2;
  output_sizes[1] = r1 * dim_1 * r2;
  output_sizes[2] = r1 * r2 * dim_2;

  uint64_t arr_O_size = output_sizes[ncm];

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  double* arr_O_1 = allocate_aligned_array(arr_O_size); 

  auto start_1 = high_resolution_clock::now();
  
  // Performing TTMC contraction using CPU - 5 for loops
  performContraction_cpu_1(mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, mode_2_ptr, mode_2_idx, 
                    values, arr_A, arr_B, arr_O_1, arr_A_size, arr_B_size, arr_O_size, ncm, l, m, n, r1, r2);

  auto end_1 = high_resolution_clock::now();
  auto duration_1 = duration_cast<microseconds>(end_1 - start_1);
  double seconds_1 = duration_1.count() / 1e6; // Convert microseconds to seconds

  // Output time taken with 2 decimal places
  cout << fixed << setprecision(2); // Set fixed-point notation and precision
  cout << "Time taken by CPU Method - 1 [5-for loop] i.e. contraction 1 : " << seconds_1 << " seconds" << endl;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Performing TTMC contraction using 4 for loops
  double* arr_O_2 = allocate_aligned_array(arr_O_size); 

  // Record start time
  auto start_2 = high_resolution_clock::now();

  
  // Perform contraction using CPU - 4 for loops
  performContraction_cpu_2(mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, mode_2_ptr, mode_2_idx, 
    values, arr_A, arr_B, arr_O_2, arr_A_size, arr_B_size, arr_O_size, ncm, l, m, n, r1, r2);
    
  // Record end time
  auto end_2 = high_resolution_clock::now();
  auto duration_2 = duration_cast<microseconds>(end_2 - start_2);
  double seconds_2 = duration_2.count() / 1e6; // Convert microseconds to seconds
  
  // Output time taken with 2 decimal places
  cout << fixed << setprecision(2); // Set fixed-point notation and precision
  cout << "Time taken by CPU Method - 2 [4-for loop] i.e. contraction 2 : " << seconds_2 << " seconds" << endl;
  
  bool correct_cpu_1_cpu_2 = compare_matrices(arr_O_1, arr_O_2, 1, arr_O_size);
  
  if (correct_cpu_1_cpu_2) {
    std::cout << "Output tensors from CPU Method-1[5-for loops] and CPU Method-2[4-for loops] are same." << std::endl;
  } else {
    std::cout << "Output tensors from CPU Method-1[5-for loops] and CPU Method-2[4-for loops] are not same." << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  double* arr_O_3 = allocate_aligned_array(arr_O_size); 

  // Record start time
  auto start_3 = high_resolution_clock::now();

  // Performing TTMC contraction using GPU - 5 for loops
  performContraction_gpu_1(mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, mode_2_ptr, mode_2_idx, 
                      values, arr_A, arr_B, arr_O_3, arr_A_size, arr_B_size, arr_O_size, ncm, l, m, n, r1, r2, total_values,
                      size_mode_0_ptr, size_mode_1_ptr, size_mode_2_ptr, size_mode_0_idx, size_mode_1_idx, size_mode_2_idx);

  // Record end time
  auto end_3 = high_resolution_clock::now();
  auto duration_3 = duration_cast<microseconds>(end_3 - start_3);
  double seconds_3 = duration_3.count() / 1e6; // Convert microseconds to seconds

  // Output time taken with 2 decimal places
  cout << fixed << setprecision(2); // Set fixed-point notation and precision
  cout << "Time taken by GPU Method - 1 [5-for loop] i.e. contraction 3 : " << seconds_3 << " seconds" << endl;

  bool correct_cpu_1_gpu_1 = compare_matrices(arr_O_1, arr_O_3, 1, arr_O_size);

  if (correct_cpu_1_gpu_1) {
      std::cout << "Output tensors from CPU Method-1[5-for loops] and GPU Method-1[5-for loops] are same." << std::endl;
  } else {
      std::cout << "Output tensors from CPU Method-1[5-for loops] and GPU Method-1[5-for loops] are not same." << std::endl;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  double* arr_O_4 =  allocate_aligned_array(arr_O_size); 
  // Record start time
  auto start_4 = high_resolution_clock::now();
  
  // Performing TTMC contraction using GPU - 4 for loops
  performContraction_gpu_2(mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, mode_2_ptr, mode_2_idx, 
    values, arr_A, arr_B, arr_O_4, arr_A_size, arr_B_size, arr_O_size, ncm, l, m, n, r1, r2, total_values,
    size_mode_0_ptr, size_mode_1_ptr, size_mode_2_ptr, size_mode_0_idx, size_mode_1_idx, size_mode_2_idx);
    
  // Record end time
  auto end_4 = high_resolution_clock::now();
  auto duration_4 = duration_cast<microseconds>(end_4 - start_4);
  double seconds_4 = duration_4.count() / 1e6; // Convert microseconds to seconds
  
  // Output time taken with 2 decimal places
  cout << fixed << setprecision(2); // Set fixed-point notation and precision
  cout << "Time taken by GPU Method - 2 [4-for loop] i.e. contraction 4 : " << seconds_4 << " seconds" << endl;
  
  bool correct_cpu_2_gpu_2 = compare_matrices(arr_O_1, arr_O_4, 1, arr_O_size);
  
  if (correct_cpu_2_gpu_2) {
    std::cout << "Output tensors from CPU Method-2[4-for loops] and GPU Method-2[4-for loops] are same." << std::endl;
  } else {
    std::cout << "Output tensors from CPU Method-2[4-for loops] and GPU Method-2[4-for loops] are not same." << std::endl;
  }
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




  delete[] my_tensor_indices; 
  delete[] my_tensor_values; 
  delete[] mode_0_ptr;
  delete[] mode_0_idx;
  delete[] mode_1_ptr;
  delete[] mode_1_idx;
  delete[] mode_2_ptr;
  delete[] mode_2_idx;

  std::free(arr_A);
  std::free(arr_B);
  std::free(arr_O_1);
  std::free(arr_O_2);
  std::free(arr_O_3);
  std::free(arr_O_4);
}