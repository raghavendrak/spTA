#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include "csf_tensor.h"
#include "matrix_utils.h"

using namespace std;

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
//////////////////////////////////////////////////////////////////// 

// Include the reference implementation for validation
#define INCLUDED_AS_LIBRARY
#include "v2_cpu_4loop.cu"

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
        } else if (arg == "-n" || arg == "--ncm" && i + 1 < argc) {
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
        double *values;
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
        double *arr_A = nullptr, *arr_B = nullptr;
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
        
        // Allocate output arrays
        double* arr_O = allocate_aligned_array(arr_O_size);
        double* ref_O = allocate_aligned_array(arr_O_size);
        
        // Run this implementation (CPU 5-loop) first
        if (verbose) {
            cout << "Running CPU 5-loop implementation..." << endl;
        }
        auto start = std::chrono::high_resolution_clock::now();
        
        performContraction_cpu_1(
            mode_0_ptr, mode_0_idx,
            mode_1_ptr, mode_1_idx,
            mode_2_ptr, mode_2_idx,
            values, arr_A, arr_B, arr_O,
            arr_A_size, arr_B_size, arr_O_size, ncm,
            tensor.dimensions[0], tensor.dimensions[1], tensor.dimensions[2], rank1, rank2
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Now run the reference implementation (CPU 4-loop) for validation
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
            tensor.dimensions[0], tensor.dimensions[1], tensor.dimensions[2], rank1, rank2
        );
        
        auto ref_end = std::chrono::high_resolution_clock::now();
        auto ref_duration = std::chrono::duration_cast<std::chrono::microseconds>(ref_end - ref_start).count();
        
        // Validate results using compare_results from matrix_utils.h
        bool valid = compare_results(arr_O, ref_O, arr_O_size);
        
        // Report results
        if (verbose) {
            cout << "CPU 5-loop execution time: " << duration / 1000.0 << " ms" << endl;
            cout << "Reference execution time: " << ref_duration / 1000.0 << " ms" << endl;
            cout << "Speedup over reference: " << (double)ref_duration / duration << "x" << endl;
            cout << "Result validation: " << (valid ? "PASSED" : "FAILED") << endl;
        } else {
            cout << "Method: CPU 5-loop, Time: " << duration / 1000.0 << " ms, Validation: " << (valid ? "PASSED" : "FAILED") << endl;
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
        free(ref_O);
        
        return valid ? 0 : 1;
    }
    catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
} 