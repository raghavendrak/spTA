#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <random>
#include <iostream>
#include <cmath>
#include <cstdlib>

// Function to generate a random matrix
void generate_matrix(uint64_t rows, uint64_t cols, unsigned int seed, float*& arr) {
  // Allocate memory for the matrix
  arr = new float[rows * cols];

  // Initialize random number generator with seed
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  // Fill matrix with random values
  for (uint64_t i = 0; i < rows * cols; i++) {
    arr[i] = dist(gen);
  }
}

// Function to compare two matrices
bool compare_results1(float*& C1, float*& C2, uint64_t size,  float tolerance = 1e-6)
{
  for (int i = 0; i < size ; ++i) {
    if (std::fabs(C1[i] - C2[i]) > tolerance) {
      std::cout << " output NOT matching at index : " << i << std::endl;
      for(int j = i; j < i + 10; ++j){
        if(j < size){
          std::cout << "C1[" << j << "] = " << C1[j] << " C2[" << j << "] = " << C2[j] << std::endl;
        }
      }
      return false;
    }
  }
  return true;
}

// Function to compare results against reference implementation
bool compare_results(float* result, float* reference, uint64_t size, float tolerance = 1e-5) {
    float max_diff = 0.0;
    float max_val = 0.0;
    int errors = 0;
    
    for (uint64_t i = 0; i < size; i++) {
        float diff = std::fabs(result[i] - reference[i]);
        max_diff = std::max(max_diff, diff);
        max_val = std::max(max_val, std::fabs(reference[i]));
        
        if (diff > tolerance && std::fabs(reference[i]) > tolerance) {
            errors++;
        }
    }
    
    float rel_error = (max_val > 0) ? max_diff / max_val : max_diff;
    
    std::cout << "Validation: Max absolute diff = " << max_diff 
         << ", Relative error = " << rel_error 
         << ", Elements with significant error = " << errors << std::endl;
         
    return (rel_error < tolerance && errors == 0);
}

// Function for aligned memory allocation
float* allocate_aligned_array(size_t num_elements) {
    constexpr size_t alignment = 32;           // 32 bytes = 256 bits
    constexpr size_t element_size = sizeof(float); // 8 bytes per float

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
    float* arr = static_cast<float*>(ptr);
    for (size_t i = 0; i < total_elements; ++i) {
      arr[i] = 0.0;
    }

    return static_cast<float*>(ptr);
}

#endif // MATRIX_UTILS_H 