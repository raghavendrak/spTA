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
void cpu_factorize_n_fuse(
    uint64_t** mode_ptrs, uint64_t** mode_idxs,
    float* values, float* factor_matrices[],
    float* arr_O, int ncm,
    uint64_t ranks[], int order, uint64_t dimensions[])
{
  if(order == 3){
    uint64_t i, j, k, index_A, index_B, index_O;
    float value;
    if(ncm == 0){

      uint64_t f1 = ranks[1];
      uint64_t f2 = ranks[2];
      float* arr_A = factor_matrices[1];
      float* arr_B = factor_matrices[2];
      float* buffer = allocate_aligned_array(f2);    // buffer for mode-s
  
      // Traverse through CSF tensor pointer and indices arrays for all modes
      for (uint64_t i_ptr = 0; i_ptr < mode_ptrs[0][1]; ++i_ptr) {
        i = mode_idxs[0][i_ptr] ;                         // Index in the mode 'i'
        
        for (uint64_t j_ptr = mode_ptrs[1][i_ptr]; j_ptr < mode_ptrs[1][i_ptr + 1]; ++j_ptr) {
          j = mode_idxs[1][j_ptr] ;                     // Index for 'j' mode in CSF
          
          memset(buffer, 0, f2 * sizeof(float));             // Set the entire memory block to 0
          
          for (uint64_t k_ptr = mode_ptrs[2][j_ptr]; k_ptr < mode_ptrs[2][j_ptr + 1]; ++k_ptr) {
            k = mode_idxs[2][k_ptr] ;                 // Index for 'k' mode in CSF
  
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
    else if(ncm == 1){
      uint64_t f1 = ranks[0];
      uint64_t f2 = ranks[2];
      float* arr_A = factor_matrices[0];
      float* arr_B = factor_matrices[2];
      
      float* buffer = allocate_aligned_array(f2);    // buffer for mode-s
      
      // Traverse through CSF tensor pointer and indices arrays for all modes
      for (uint64_t i_ptr = 0; i_ptr < mode_ptrs[0][1]; ++i_ptr) {
        i = mode_idxs[0][i_ptr] ;                         // Index in the mode 'i'
  
        for (uint64_t j_ptr = mode_ptrs[1][i_ptr]; j_ptr < mode_ptrs[1][i_ptr + 1]; ++j_ptr) {
          j = mode_idxs[1][j_ptr] ;                     // Index for 'j' mode in CSF
  
          memset(buffer, 0, f2 * sizeof(float));             // Set the entire memory block to 0
  
          for (uint64_t k_ptr = mode_ptrs[2][j_ptr]; k_ptr < mode_ptrs[2][j_ptr + 1]; ++k_ptr) {
            k = mode_idxs[2][k_ptr] ;                 // Index for 'k' mode in CSF
  
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
    else if(ncm == 2){
      uint64_t f1 = ranks[0];
      uint64_t f2 = ranks[1];
      float* arr_A = factor_matrices[0];
      float* arr_B = factor_matrices[1];
      uint64_t n = dimensions[ncm];
      float* buffer = allocate_aligned_array(n*f2);    // buffer for mode-k and mode-s
      bool* k_buffer = new bool[n];  // buffer for k-indices
      uint64_t index_buf = 0;
  
      // Traverse through CSF tensor pointer and indices arrays for all modes
      for (uint64_t i_ptr = 0; i_ptr < mode_ptrs[0][1]; ++i_ptr) {
        i = mode_idxs[0][i_ptr] ;                          // Index in the mode 'i'
  
        memset(buffer, 0, n * f2 * sizeof(float));             // Set the entire memory block to 0
        memset(k_buffer, 0, n * sizeof(bool)); //initialize to false
        for (uint64_t j_ptr = mode_ptrs[1][i_ptr]; j_ptr < mode_ptrs[1][i_ptr + 1]; ++j_ptr) {
          j = mode_idxs[1][j_ptr] ;                      // Index for 'j' mode in CSF
  
          for (uint64_t k_ptr = mode_ptrs[2][j_ptr]; k_ptr < mode_ptrs[2][j_ptr + 1]; ++k_ptr) {
            k = mode_idxs[2][k_ptr] ;                  // Index for 'k' mode in CSF
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
  else if(order == 4){
    uint64_t i, j, k, l;
    float value;
    float* buf1 = allocate_aligned_array(ranks[3]);
    float* buf2 = allocate_aligned_array(ranks[2] * ranks[3]);
    if(ncm == 0){
      // Traverse through CSF tensor pointer and indices arrays for all modes
      for (uint64_t i_ptr = 0; i_ptr < mode_ptrs[0][1]; ++i_ptr) {
        i = mode_idxs[0][i_ptr];                         // Index in the mode 'i'
  
        for (uint64_t j_ptr = mode_ptrs[1][i_ptr]; j_ptr < mode_ptrs[1][i_ptr + 1]; ++j_ptr) {
          j = mode_idxs[1][j_ptr];                     // Index for 'j' mode in CSF
          memset(buf2, 0, ranks[2] * ranks[3] * sizeof(float));
          
          for (uint64_t k_ptr = mode_ptrs[2][j_ptr]; k_ptr < mode_ptrs[2][j_ptr + 1]; ++k_ptr) {
            k = mode_idxs[2][k_ptr];                 // Index for 'k' mode in CSF
            memset(buf1, 0, ranks[3] * sizeof(float));

            for (uint64_t l_ptr = mode_ptrs[3][k_ptr]; l_ptr < mode_ptrs[3][k_ptr + 1]; ++l_ptr) {
              l = mode_idxs[3][l_ptr];                 // Index for 'l' mode in CSF
              value = values[l_ptr];                  // CSF value for the above i, j, k, l

              for(uint64_t t = 0; t < ranks[3]; ++t){
                buf1[t] += value * factor_matrices[3][l * ranks[3] + t];
              }
            }

            for(uint64_t s = 0; s < ranks[2]; ++s){
              for(uint64_t t = 0; t < ranks[3]; ++t){
                buf2[s * ranks[3] + t] += buf1[t] * factor_matrices[2][k * ranks[2] + s];
              }
            }
          }

          for(uint64_t r = 0; r < ranks[1]; ++r){
            for(uint64_t s = 0; s < ranks[2]; ++s){
              for(uint64_t t = 0; t < ranks[3]; ++t){
                arr_O[i * ranks[1] * ranks[2] * ranks[3] + 
                      r * ranks[2] * ranks[3] + 
                      s * ranks[3] + 
                      t] 
                      += 
                      buf2[s * ranks[3] + t] * 
                      factor_matrices[1][j * ranks[1] + r];
              }
            }
          }
        }
      }
    }
    std::free(buf1);
    std::free(buf2);
  }
}

/*End of CPU 4 loop Method*/
//////////////////////////////////////////////////////////////////// 

// Only compile the main function when this file is compiled directly, not when included
#ifndef INCLUDED_AS_LIBRARY

int main(int argc, char* argv[]) {
    bool verbose = false;
    string csf_file;
    std::vector<uint64_t> ranks;
    int ncm = 0;
    
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
        
        // Run this implementation (CPU factorize-n-fuse)
        if (verbose) {
            cout << "Running CPU factorize and fuse implementation..." << endl;
        }
        auto start = std::chrono::high_resolution_clock::now();
        
        cpu_factorize_n_fuse(
          mode_ptrs.data(), mode_idxs.data(),
            values, factor_matrices.data(), arr_O,
            ncm, ranks.data(), order, tensor.dimensions.data()
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Report results
        if (verbose) {
            cout << "Method: CPU_FnF, Time: " << duration / 1000.0 << " ms" << endl;
        } else {
            cout << "Method: CPU_FnF, Time: " << duration / 1000.0 << " ms" << endl;
        }
        
        // Clean up
        for(int i = 0; i < order; i++){
          free(factor_matrices[i]);
        }
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