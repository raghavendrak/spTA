#ifndef V2_CPU_4LOOP_INCLUDED
#define V2_CPU_4LOOP_INCLUDED

#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include "csf_tensor.h"
#include "matrix_utils.h"

using namespace std;

/////////////////////////////////////////////////////////////////////
/*Start of CPU 4 loop Method*/

// Function to perform contraction based on the inputs using 4 for loops
void performContraction_cpu_2(uint64_t*& mode_0_ptr,uint64_t*& mode_0_idx,
                        uint64_t*& mode_1_ptr, uint64_t*& mode_1_idx,
                        uint64_t*& mode_2_ptr, uint64_t*& mode_2_idx,
                        float*& values, float*& arr_A, float*& arr_B,  
                        float*& arr_O, uint64_t& arr_A_size, uint64_t& arr_B_size, uint64_t& arr_O_size, int& contraction, 
                        uint64_t& l, uint64_t& m, uint64_t& n, uint64_t& f1, uint64_t& f2) 
{
  uint64_t i, j, k, index_A, index_B, index_O;
  float value;
                            
  if(contraction == 0){
    float* buffer = allocate_aligned_array(f2);    // buffer for mode-s

    // Traverse through CSF tensor pointer and indices arrays for all modes
    for (uint64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
      i = mode_0_idx[i_ptr] ;                         // Index in the mode 'i'
      
      for (uint64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
        j = mode_1_idx[j_ptr] ;                     // Index for 'j' mode in CSF
        
        memset(buffer, 0, f2 * sizeof(float));             // Set the entire memory block to 0
        
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
    float* buffer = allocate_aligned_array(f2);    // buffer for mode-s
    
    // Traverse through CSF tensor pointer and indices arrays for all modes
    for (uint64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
      i = mode_0_idx[i_ptr] ;                         // Index in the mode 'i'

      for (uint64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
        j = mode_1_idx[j_ptr] ;                     // Index for 'j' mode in CSF

        memset(buffer, 0, f2 * sizeof(float));             // Set the entire memory block to 0

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
    float* buffer = allocate_aligned_array(n*f2);    // buffer for mode-k and mode-s
    bool* k_buffer = new bool[n];  // buffer for k-indices
    uint64_t index_buf = 0;

    // Traverse through CSF tensor pointer and indices arrays for all modes
    for (uint64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
      i = mode_0_idx[i_ptr] ;                          // Index in the mode 'i'

      memset(buffer, 0, n * f2 * sizeof(float));             // Set the entire memory block to 0
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
//////////////////////////////////////////////////////////////////// 

// Only compile the main function when this file is compiled directly, not when included
#ifndef INCLUDED_AS_LIBRARY

int main(int argc, char* argv[]) {
    bool verbose = false;
    string csf_file;
    uint64_t rank1 = 30, rank2 = 30;
    int ncm = 0;
    
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
        float *values;
        int order;
        
        getCSFArrays(tensor, &mode_0_ptr, &mode_0_idx, 
                    &mode_1_ptr, &mode_1_idx, 
                    &mode_2_ptr, &mode_2_idx, 
                    &values, &order);
        
        // Calculate matrix dimensions based on contraction mode
        uint64_t matrix_dim1 = getMatrixDim1(tensor.dimensions, ncm);
        uint64_t matrix_dim2 = getMatrixDim2(tensor.dimensions, ncm);
        uint64_t out_dim1 = getOutputDim1(tensor.dimensions, ncm);
        
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
            cout << "Output dimensions: " << out_dim1 << " x " << out_dim2 << endl;
        }
        
        // Allocate output array
        float* arr_O = allocate_aligned_array(arr_O_size);
        
        // Run this implementation (CPU 4-loop - reference)
        if (verbose) {
            cout << "Running CPU 4-loop implementation..." << endl;
        }
        auto start = std::chrono::high_resolution_clock::now();
        
        performContraction_cpu_2(
            mode_0_ptr, mode_0_idx,
            mode_1_ptr, mode_1_idx,
            mode_2_ptr, mode_2_idx,
            values, arr_A, arr_B, arr_O,
            arr_A_size, arr_B_size, arr_O_size, ncm,
            tensor.dimensions[0], tensor.dimensions[1], tensor.dimensions[2], rank1, rank2
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Report results
        if (verbose) {
            cout << "Method: CPU_4L, Time: " << duration / 1000.0 << " ms" << endl;
        } else {
            cout << "Method: CPU_4L, Time: " << duration / 1000.0 << " ms" << endl;
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
        
        return 0;
    }
    catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}

#endif // INCLUDED_AS_LIBRARY

#endif // V2_CPU_4LOOP_INCLUDED 