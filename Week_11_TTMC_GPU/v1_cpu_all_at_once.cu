#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include "csf_tensor.h"
#include "matrix_utils.h"

using namespace std;

/////////////////////////////////////////////////////////////////////
/*Start of CPU All At Once Method*/
void cpu_all_at_once(uint64_t** mode_ptrs, uint64_t** mode_idxs,
                        float*& values, float* factor_matrices[],  
                        float*& arr_O, int& ncm, 
                        uint64_t ranks[], int order) 
{ 
  if(order == 3){
    uint64_t i, j, k, index_A, index_B, index_O;
    float value;
    if(ncm == 0){
      uint64_t f1 = ranks[1];
      uint64_t f2 = ranks[2];
      float* arr_A = factor_matrices[1];
      float* arr_B = factor_matrices[2];
      // Traverse through CSF tensor pointer and indices arrays for all modes
      for (uint64_t i_ptr = 0; i_ptr < mode_ptrs[0][1]; ++i_ptr) {
        i = mode_idxs[0][i_ptr];                         // Index in the mode 'i'
  
        for (uint64_t j_ptr = mode_ptrs[1][i_ptr]; j_ptr < mode_ptrs[1][i_ptr + 1]; ++j_ptr) {
          j = mode_idxs[1][j_ptr];                     // Index for 'j' mode in CSF
  
          for (uint64_t k_ptr = mode_ptrs[2][j_ptr]; k_ptr < mode_ptrs[2][j_ptr + 1]; ++k_ptr) {
            k = mode_idxs[2][k_ptr];                 // Index for 'k' mode in CSF
  
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
    else if(ncm == 1){
      uint64_t f1 = ranks[0];
      uint64_t f2 = ranks[2];
      float* arr_A = factor_matrices[0];
      float* arr_B = factor_matrices[2];
      // Traverse through CSF tensor pointer and indices arrays for all modes
      for (uint64_t i_ptr = 0; i_ptr < mode_ptrs[0][1]; ++i_ptr) {
        i = mode_idxs[0][i_ptr];                         // Index in the mode 'i'
  
        for (uint64_t j_ptr = mode_ptrs[1][i_ptr]; j_ptr < mode_ptrs[1][i_ptr + 1]; ++j_ptr) {
          j = mode_idxs[1][j_ptr];                     // Index for 'j' mode in CSF
  
          for (uint64_t k_ptr = mode_ptrs[2][j_ptr]; k_ptr < mode_ptrs[2][j_ptr + 1]; ++k_ptr) {
            k = mode_idxs[2][k_ptr];                 // Index for 'k' mode in CSF
  
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
    else if(ncm == 2){
      uint64_t f1 = ranks[0];
      uint64_t f2 = ranks[1];
      float* arr_A = factor_matrices[0];
      float* arr_B = factor_matrices[1];
      // Traverse through CSF tensor pointer and indices arrays for all modes
      for (uint64_t i_ptr = 0; i_ptr < mode_ptrs[0][1]; ++i_ptr) {
        i = mode_idxs[0][i_ptr] ;                         // Index in the mode 'i'
  
        for (uint64_t j_ptr = mode_ptrs[1][i_ptr]; j_ptr < mode_ptrs[1][i_ptr + 1]; ++j_ptr) {
          j = mode_idxs[1][j_ptr] ;                     // Index for 'j' mode in CSF
  
          for (uint64_t k_ptr = mode_ptrs[2][j_ptr]; k_ptr < mode_ptrs[2][j_ptr + 1]; ++k_ptr) {
            k = mode_idxs[2][k_ptr] ;                 // Index for 'k' mode in CSF
  
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
  else if(order == 4){
    
  }
}
/*End of CPU All At Once Method*/
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
        
        // Run this implementation (CPU All At Once) first
        if (verbose) {
            cout << "Running CPU All At Once implementation..." << endl;
        }
        auto start = std::chrono::high_resolution_clock::now();
        
        cpu_all_at_once(
            mode_ptrs.data(), mode_idxs.data(),
            values, factor_matrices.data(), arr_O,
            ncm, ranks.data(), order
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
            
            cpu_factorize_n_fuse(
                mode_ptrs.data(), mode_idxs.data(),
                values, factor_matrices.data(), ref_O,
                ncm, ranks.data(), order, tensor.dimensions.data() 
            );
            
            auto ref_end = std::chrono::high_resolution_clock::now();
            ref_duration = std::chrono::duration_cast<std::chrono::microseconds>(ref_end - ref_start).count();
            
            // Validate results using compare_results from matrix_utils.h
            valid = compare_results(arr_O, ref_O, arr_O_size);
        }
        
        // Report results
        if (verbose) {
            cout << "CPU All At Once execution time: " << duration / 1000.0 << " ms" << endl;
            if (verify) {
                cout << "Reference execution time: " << ref_duration / 1000.0 << " ms" << endl;
                cout << "Speedup over reference: " << (float)ref_duration / duration << "x" << endl;
                cout << "Result validation: " << (valid ? "PASSED" : "FAILED") << endl;
            }
        } else {
            if (verify) {
                cout << "Method: CPU_AAO, Time: " << duration / 1000.0 << " ms, Validation: " << (valid ? "PASSED" : "FAILED") << endl;
            } else {
                cout << "Method: CPU_AAO, Time: " << duration / 1000.0 << " ms" << endl;
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