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

#include <cstdlib>  // for std::aligned_alloc, std::free, size_t
#include <cstring>  // for std::memset
#include <stdexcept> // for std::runtime_error

using namespace std;
using namespace std::chrono;

// Struct to hold CSF tensor data (from CSF_reader.cpp)
struct CSFTensor {
    int order;  // Number of modes
    vector<vector<uint64_t>> ptrs;  // Pointers for each mode
    vector<vector<uint64_t>> idxs;  // Indices for each mode
    vector<double> values;  // Non-zero values
    vector<uint64_t> dimensions; // Dimensions of each mode
    vector<int> modeOrdering; // Original mode ordering
    
    // Initialize with given order
    CSFTensor(int n) : order(n), ptrs(n), idxs(n), dimensions(n), modeOrdering(n) {
        for (int i = 0; i < n; i++) {
            modeOrdering[i] = i; // Default ordering
        }
    }
};

// Function to read a CSF tensor from file (from CSF_reader.cpp)
CSFTensor readCSFTensor(const string& filename) {
    ifstream inFile(filename);
    if (!inFile) {
        throw runtime_error("Unable to open input file: " + filename);
    }
    
    string line;
    int order = 0;
    CSFTensor* tensor = nullptr;
    
    // Read each line
    while (getline(inFile, line)) {
        // Skip empty lines
        if (line.empty()) continue;
        
        // Handle comments (detect tensor order, mode ordering, and dimensions if present)
        if (line[0] == '#') {
            if (line.find("Tensor order:") != string::npos) {
                istringstream iss(line.substr(line.find(":") + 1));
                iss >> order;
                cout << "Found tensor order: " << order << endl;
                tensor = new CSFTensor(order);
            }
            else if (line.find("Mode ordering:") != string::npos) {
                if (tensor == nullptr) {
                    cerr << "Error: Mode ordering found before tensor order" << endl;
                    throw runtime_error("Invalid CSF file format");
                }
                istringstream iss(line.substr(line.find(":") + 1));
                int mode;
                int idx = 0;
                while (iss >> mode && idx < tensor->order) {
                    tensor->modeOrdering[idx++] = mode;
                }
                cout << "Found mode ordering: ";
                for (int i = 0; i < tensor->order; i++) {
                    cout << tensor->modeOrdering[i] << " ";
                }
                cout << endl;
            }
            else if (line.find("Dimensions:") != string::npos) {
                if (tensor == nullptr) {
                    cerr << "Error: Dimensions found before tensor order" << endl;
                    throw runtime_error("Invalid CSF file format");
                }
                istringstream iss(line.substr(line.find(":") + 1));
                uint64_t dim;
                int idx = 0;
                while (iss >> dim && idx < tensor->order) {
                    tensor->dimensions[idx++] = dim;
                }
                cout << "Found tensor dimensions: ";
                for (int i = 0; i < tensor->order; i++) {
                    cout << tensor->dimensions[i] << " ";
                }
                cout << endl;
            }
            continue;
        }
        
        // Parse data lines
        istringstream iss(line);
        string label;
        
        // Read until the colon
        getline(iss, label, ':');
        
        // If we haven't found the order yet, try to determine it
        if (tensor == nullptr) {
            if (label.find("mode_") == 0 && label.find("_ptr") != string::npos) {
                int modeIdx = stoi(label.substr(5, label.find("_ptr") - 5));
                order = max(order, modeIdx + 1);
            }
        }
        
        // Ensure we have a tensor object
        if (tensor == nullptr && order > 0) {
            tensor = new CSFTensor(order);
        } else if (tensor == nullptr) {
            throw runtime_error("Could not determine tensor order from file");
        }
        
        // Parse based on label
        if (label.find("mode_") == 0) {
            int modeIdx = stoi(label.substr(5, label.find("_", 5) - 5));
            
            if (label.find("_ptr") != string::npos) {
                // Parse mode pointers
                uint64_t val;
                while (iss >> val) {
                    tensor->ptrs[modeIdx].push_back(val);
                }
                //cout << "Read " << tensor->ptrs[modeIdx].size() << " pointers for mode " << modeIdx << endl;
            } else if (label.find("_idx") != string::npos) {
                // Parse mode indices
                uint64_t val;
                while (iss >> val) {
                    tensor->idxs[modeIdx].push_back(val);
                }
                //cout << "Read " << tensor->idxs[modeIdx].size() << " indices for mode " << modeIdx << endl;
            }
        } else if (label == "values") {
            // Parse values
            double val;
            while (iss >> val) {
                tensor->values.push_back(val);
            }
            //cout << "Read " << tensor->values.size() << " non-zero values" << endl;
        }
    }
    
    inFile.close();
    
    if (tensor == nullptr) {
        throw runtime_error("Failed to parse CSF tensor from file");
    }
    
    CSFTensor result = *tensor;
    delete tensor;
    return result;
}

// Modified version of getCSFArrays to handle separate arrays for each mode
void getCSFArrays(const CSFTensor& tensor, 
                 uint64_t** mode_0_ptr, uint64_t** mode_0_idx,
                 uint64_t** mode_1_ptr, uint64_t** mode_1_idx,
                 uint64_t** mode_2_ptr, uint64_t** mode_2_idx,
                 double** values, int* order) {
    // Set the order
    *order = tensor.order;
    
    if (tensor.order != 3) {
        throw runtime_error("Only 3rd order tensors are supported");
    }
    
    // Copy pointers and indices for each mode
    *mode_0_ptr = new uint64_t[tensor.ptrs[0].size()];
    *mode_0_idx = new uint64_t[tensor.idxs[0].size()];
    *mode_1_ptr = new uint64_t[tensor.ptrs[1].size()];
    *mode_1_idx = new uint64_t[tensor.idxs[1].size()];
    *mode_2_ptr = new uint64_t[tensor.ptrs[2].size()];
    *mode_2_idx = new uint64_t[tensor.idxs[2].size()];
    
    // Copy data for mode 0
    for (size_t i = 0; i < tensor.ptrs[0].size(); i++) {
        (*mode_0_ptr)[i] = tensor.ptrs[0][i];
    }
    for (size_t i = 0; i < tensor.idxs[0].size(); i++) {
        (*mode_0_idx)[i] = tensor.idxs[0][i];
    }
    
    // Copy data for mode 1
    for (size_t i = 0; i < tensor.ptrs[1].size(); i++) {
        (*mode_1_ptr)[i] = tensor.ptrs[1][i];
    }
    for (size_t i = 0; i < tensor.idxs[1].size(); i++) {
        (*mode_1_idx)[i] = tensor.idxs[1][i];
    }
    
    // Copy data for mode 2
    for (size_t i = 0; i < tensor.ptrs[2].size(); i++) {
        (*mode_2_ptr)[i] = tensor.ptrs[2][i];
    }
    for (size_t i = 0; i < tensor.idxs[2].size(); i++) {
        (*mode_2_idx)[i] = tensor.idxs[2][i];
    }
    
    // Copy values
    *values = new double[tensor.values.size()];
    for (size_t i = 0; i < tensor.values.size(); i++) {
        (*values)[i] = tensor.values[i];
    }    
    
}

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


/////////////////////////////////////////////////////
void generate_matrix( uint64_t rows, uint64_t cols, unsigned int seed,  double*& arr) {
  // Allocate memory for the matrix
  arr = new double[rows * cols];

  // Initialize random number generator with seed
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  // Fill matrix with random values
  for (uint64_t i = 0; i < rows * cols; i++) {
    arr[i] = dist(gen);
  }
}
/////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Start of Function for comparing the results of various contractions*/
bool compare_matrices(double*& C1, double*& C2, int rows, int cols, double tolerance = 1e-6)
{
  for (int i = 0; i < rows * cols; ++i) {
    if (std::fabs(C1[i] - C2[i]) > tolerance) {
      std::cout << " NOT matching at i : " << i << endl;
      for(int j = i; j < i + 10; ++j){
        if(j < rows * cols){
          std::cout << "C1[i] = " << C1[j] << " C2[i] = " << C2[j] << endl;
        }
      }
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
          if(blockIdx.x == 0 && threadIdx.x == 0){
            printf("Memory allocation failure \n");
          }
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

  // Launch kernel
  int threadsPerBlock = 256;
  
  // parallelising 'j_ptr' :
  int blocksPerGrid = (size_mode_1_idx + threadsPerBlock - 1) / threadsPerBlock;
  
  if(contraction == 0 || contraction == 1){
    // parallelising 'j_ptr' for contraction = 0 and contraction = 1 :
    cudaMalloc(&buffer_for_contraction_0_1, f2 * size_mode_1_idx * sizeof(double));
    // parallelising 'j_ptr' for contraction = 0 and contraction = 1 :
    cudaMemset(buffer_for_contraction_0_1, 0, f2 * size_mode_1_idx * sizeof(double));
    
    // parallelising 'i_ptr' :
    contractionKernel_4<<<blocksPerGrid, threadsPerBlock>>>(
      d_mode_0_ptr, d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
      d_values, d_arr_A, d_arr_B, d_arr_O, l, m, n, f1, f2, contraction, buffer_for_contraction_0_1);
    }
    else if(contraction == 2){
      // parallelising 'j_ptr' for contraction = 2 :
      cudaMalloc(&buffer_for_contraction_2, n * f2 * size_mode_1_idx * sizeof(double));
      cudaMalloc(&k_buffer_for_contraction_2, n * size_mode_1_idx * sizeof(int));
      
      // parallelising 'j_ptr' for contraction = 2 :
      cudaMemset(buffer_for_contraction_2, 0, n * f2 * size_mode_1_idx * sizeof(double));
      cudaMemset(k_buffer_for_contraction_2, 0, n * size_mode_1_idx * sizeof(int));

      contractionKernel_for_second_contraction_part_1<<<blocksPerGrid, threadsPerBlock>>>(
        d_mode_0_ptr, d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
        d_values, d_arr_A, d_arr_B, d_arr_O, l, m, n, f1, f2, contraction, buffer_for_contraction_2, k_buffer_for_contraction_2);
        cudaDeviceSynchronize();
        contractionKernel_for_second_contraction_part_2<<<blocksPerGrid, threadsPerBlock>>>(
          d_mode_0_ptr, d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
          d_values, d_arr_A, d_arr_B, d_arr_O, l, m, n, f1, f2, contraction, buffer_for_contraction_2, k_buffer_for_contraction_2);
    }
        
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

  cudaFree(buffer_for_contraction_0_1);
  cudaFree(buffer_for_contraction_2);
  cudaFree(k_buffer_for_contraction_2);
  // cudaDeviceSynchronize();
}

/*End of host function for GPU 4 loop Method*/
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/*start of device function for GPU 4 loop Method using STREAMS*/
__global__ void GPU_4loop_streams(
  // uint64_t* mode_1_ptr,
  uint64_t* mode_1_idx,
  uint64_t* mode_2_ptr, uint64_t* mode_2_idx,
  double* values, double* arr_A, double* arr_B,  
  double* arr_O, uint64_t l, uint64_t m, uint64_t n, uint64_t f1, uint64_t f2, int ncm,
  int size_mode_0_ptr, int size_mode_1_ptr, int size_mode_2_ptr,
  int size_mode_0_idx, int size_mode_1_idx, int size_mode_2_idx, uint64_t i, uint64_t j_ptr_offset
)
{
  extern __shared__ double buf[];
  uint64_t j, j_ptr, k, k_ptr, k_ptr_offset, index_A, index_B, index_O ;
  int r, s, r_offset, s_offset, WARP_SIZE = 32;
  double value, A_val;
  unsigned mask;

  j_ptr = j_ptr_offset + blockIdx.x;
  j = mode_1_idx[j_ptr];
  // uint64_t nnz_k = mode_2_ptr[j_ptr+1] - mode_2_ptr[j_ptr];
  
  int buf_index = threadIdx.y * blockDim.x + threadIdx.x;

  //NOTE; WORKS ONLY IF f2 < 1024
  if(buf_index < f2){
    buf[buf_index] = 0.0;
  }
  __syncthreads();
  
  // parallelize k across warps
  // block dimesion is 32 x 32. 
  // hence, each row of thread block will form a warp 
  // each row of thread block(a warp) picks a k, thus a nonzero of input tensor
  for(k_ptr_offset = mode_2_ptr[j_ptr]; k_ptr_offset < mode_2_ptr[j_ptr + 1]; k_ptr_offset += blockDim.y){
    k_ptr =  k_ptr_offset + threadIdx.y;
    if(k_ptr < mode_2_ptr[j_ptr + 1]){
      
      value = values[k_ptr];
      k = mode_2_idx[k_ptr];
      
      //Each thread in a warp picks a 's'
      for(s_offset = 0; s_offset < f2; s_offset += blockDim.x){
        s = s_offset + threadIdx.x;
        if(s < f2){
          // mask = __activemask();
          index_B = k * f2 + s;
          // double prod_val = value * arr_B[index_B];

          // for(int shuffle_offset = WARP_SIZE/2; shuffle_offset > 0; shuffle_offset>>=1){
          //   prod_val += __shfl_down_sync(mask, prod_val, shuffle_offset);
          // }
          // if(threadIdx.x == 0) buf[s] += prod_val;
          atomicAdd(&buf[s], value * arr_B[index_B] );
        }
      }
    }
  }
  __syncthreads();
  
  //////////////////////////////////////////////////////////////////////////////////
  // parallelize 'r' across warps
  // block dimesion is 32 x 32. 
  // hence, each row of thread block will form a warp 
  // each row of thread block(a warp) picks a 'r'
  if(ncm == 0){
    for(r_offset = 0; r_offset < f1; r_offset += blockDim.y){
      r = r_offset + threadIdx.y;
      if(r < f1){
        index_A = j * f1 + r;
        A_val = arr_A[index_A];
        //Each thread in a warp picks a 's'
        for(s_offset = 0; s_offset < f2; s_offset += blockDim.x){
          s = s_offset + threadIdx.x;
          if(s < f2){
            index_O = i * f1 * f2 + r * f2  + s;
            //atomic add is required since different threadblocks in the same stream has same i
            atomicAdd(&arr_O[index_O], buf[s] * A_val);
          }
        }
        
      }
    }
  }
  else if(ncm == 1){
    for(r_offset = 0; r_offset < f1; r_offset += blockDim.y){
      r = r_offset + threadIdx.y;
      if(r < f1){
        index_A = i * f1 + r;
        A_val = arr_A[index_A];
        //Each thread in a warp picks a 's'
        for(s_offset = 0; s_offset < f2; s_offset += blockDim.x){
          s = s_offset + threadIdx.x;
          if(s < f2){
            index_O = j * f1 * f2 + r * f2  + s;
            //atomic add is required since different threadblocks in the same stream has same i
            atomicAdd(&arr_O[index_O], buf[s] * A_val);
          }
        }
        
      }
    }
  }
  // __syncthreads();
}

__global__ void GPU_4loop_streams_ncm_2_part_1(
  // uint64_t* mode_1_ptr,
  uint64_t* mode_1_idx,
  uint64_t* mode_2_ptr, uint64_t* mode_2_idx,
  double* values, double* arr_A, double* arr_B,  
  double* arr_O, uint64_t l, uint64_t m, uint64_t n, uint64_t f1, uint64_t f2, int ncm,
  int size_mode_0_ptr, int size_mode_1_ptr, int size_mode_2_ptr,
  int size_mode_0_idx, int size_mode_1_idx, int size_mode_2_idx, uint64_t i, uint64_t j_ptr_offset,
  double* buffer_for_ncm_2, bool* k_index_buffer
)
{ 
  //shared memory will not be enough for 2d dense buf[k,s] of type double
  // for e.g. dim_k = 1024, dim_s = 32, the required memory is 32*8*1024 = 256kb
  uint64_t j, j_ptr, k, k_ptr, k_ptr_offset, index_B ;
  int  s, s_offset, buf_index;// WARP_SIZE = 32;
  double value;
  // unsigned mask;

  j_ptr = j_ptr_offset + blockIdx.x;
  j = mode_1_idx[j_ptr];
  
  // parallelize s across warps
  // block dimesion is 32 x 32. 
  // hence, each row of thread block will form a warp 
  // each column of thread block(a warp) picks a k, thus a nonzero of input tensor
  for(k_ptr_offset = mode_2_ptr[j_ptr]; k_ptr_offset < mode_2_ptr[j_ptr + 1]; k_ptr_offset += blockDim.x){
    k_ptr =  k_ptr_offset + threadIdx.x;
    if(k_ptr < mode_2_ptr[j_ptr + 1]){
      
      value = values[k_ptr];
      k = mode_2_idx[k_ptr];
      //since each column in threadblock has same k, threads from first column is enough to note that k
      if(threadIdx.y == 0) k_index_buffer[k] = true;
      
      //Each thread in a warp picks a 's'
      for(s_offset = 0; s_offset < f2; s_offset += blockDim.y){
        s = s_offset + threadIdx.y;
        if(s < f2){
          index_B = j * f2 + s;
          buf_index = k * f2 + s;
          double prod_val = value * arr_B[index_B];
          
          //warp shuffle cannot be used here because either k or s is changing along the both block dimension
          // mask = __activemask();
          // for(int shuffle_offset = WARP_SIZE/2; shuffle_offset > 0; shuffle_offset>>=1){
          //   prod_val += __shfl_down_sync(mask, prod_val, shuffle_offset);
          // }
          // if(threadIdx.x == 0) 
          atomicAdd(&buffer_for_ncm_2[buf_index], prod_val);
        }
      }
    }
  }
  // __syncthreads(); won't work because synchronization across blocks is required 
  
  //////////////////////////////////////////////////////////////////////////////////
  
}

// __global__ void pick_non_zero_Ks(bool* k_index_buffer, uint64_t* output_indices, int N) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= N) return;
//     if(threadIdx.x == 0 && blockIdx.x == 0){
//       __shared__ uint64_t counter = 0;
//     }
//     __syncthreads();

//     if (k_index_buffer[idx]) {
//       // Atomically get next free slot
//       int out_idx = atomicAdd(counter, 1);
//       output_indices[out_idx] = idx;
//     }
// }

__global__ void GPU_4loop_streams_ncm_2_part_2(
  uint64_t* mode_1_idx,
  uint64_t* mode_2_ptr, uint64_t* mode_2_idx,
  double* values, double* arr_A, double* arr_B,  
  double* arr_O, uint64_t l, uint64_t m, uint64_t n, uint64_t f1, uint64_t f2, int ncm,
  int size_mode_0_ptr, int size_mode_1_ptr, int size_mode_2_ptr,
  int size_mode_0_idx, int size_mode_1_idx, int size_mode_2_idx, uint64_t i, uint64_t j_ptr_offset,
  double* buffer_for_ncm_2, bool* k_index_buffer
)
{
  uint64_t  k,  index_A, index_O ;
  int r, s, r_offset, s_offset, buf_index;
  double  A_val;
  k = blockIdx.x;
  if(k_index_buffer[k]){
    // parallelize 'r' across warps
    // block dimesion is 32 x 32. 
    // hence, each row of thread block will form a warp 
    // each row of thread block(a warp) picks a 'r'

    // if(threadIdx.x == 0 && threadIdx.y == 0) printf("k = %d", k);
    for(r_offset = 0; r_offset < f1; r_offset += blockDim.y){
      r = r_offset + threadIdx.y;
      if(r < f1){
        index_A = i * f1 + r;
        A_val = arr_A[index_A];
        //Each thread in a warp picks a 's'
        for(s_offset = 0; s_offset < f2; s_offset += blockDim.x){
          s = s_offset + threadIdx.x;
          if(s < f2){
            index_O = k * f1 * f2 + r * f2  + s;
            buf_index = k * f2 + s;
            //atomic add is required since different threadblocks in the same stream has same i
            atomicAdd(&arr_O[index_O], buffer_for_ncm_2[buf_index] * A_val);
          }
        }
      }
    }
  }
}
/*End of device function for GPU 4 loop Method using STREAMS*/
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/*Start of host function for GPU 4 loop Method using STREAMS*/
void GPU_4loop_host_func(
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
    // uint64_t *d_mode_0_ptr, *d_mode_0_idx, *d_mode_1_ptr;
    uint64_t *d_mode_1_idx, *d_mode_2_ptr, *d_mode_2_idx;
    double *d_values, *d_arr_A, *d_arr_B, *d_arr_O;
    // double* buffer_for_contraction_0_1;
    // double* buffer_for_contraction_2;
    // int* k_buffer_for_contraction_2;
  
    // cudaMalloc(&d_mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr);
    // cudaMalloc(&d_mode_0_idx, sizeof(uint64_t) * size_mode_0_idx);
    // cudaMalloc(&d_mode_1_ptr, sizeof(uint64_t) * size_mode_1_ptr);
    cudaMalloc(&d_mode_1_idx, sizeof(uint64_t) * size_mode_1_idx);
    cudaMalloc(&d_mode_2_ptr, sizeof(uint64_t) * size_mode_2_ptr);
    cudaMalloc(&d_mode_2_idx, sizeof(uint64_t) * size_mode_2_idx);
    cudaMalloc(&d_values, sizeof(double) * total_values);
    cudaMalloc(&d_arr_A, sizeof(double) * arr_A_size);
    cudaMalloc(&d_arr_B, sizeof(double) * arr_B_size);
    cudaMalloc(&d_arr_O, sizeof(double) * arr_O_size);
  
  
    // // parallelising 'j_ptr' for contraction = 0 and contraction = 1 :
    // cudaMalloc(&buffer_for_contraction_0_1, f2 * size_mode_1_idx * sizeof(double));
  
    // // parallelising 'j_ptr' for contraction = 2 :
    // cudaMalloc(&buffer_for_contraction_2, n * f2 * size_mode_1_idx * sizeof(double));
    // cudaMalloc(&k_buffer_for_contraction_2, n * size_mode_1_idx * sizeof(int));
  
    // Copy data to device
    // cudaMemcpy(d_mode_0_ptr, mode_0_ptr, sizeof(uint64_t) * size_mode_0_ptr, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_mode_0_idx, mode_0_idx, sizeof(uint64_t) * size_mode_0_idx, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_mode_1_ptr, mode_1_ptr, sizeof(uint64_t) * size_mode_1_ptr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_1_idx, mode_1_idx, sizeof(uint64_t) * size_mode_1_idx, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_2_ptr, mode_2_ptr, sizeof(uint64_t) * size_mode_2_ptr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_2_idx, mode_2_idx, sizeof(uint64_t) * size_mode_2_idx, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, sizeof(double) * total_values, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_A, arr_A, sizeof(double) * arr_A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_B, arr_B, sizeof(double) * arr_B_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_O, arr_O, sizeof(double) * arr_O_size, cudaMemcpyHostToDevice);
    
    
    // // parallelising 'j_ptr' for contraction = 0 and contraction = 1 :
    // cudaMemset(buffer_for_contraction_0_1, 0, f2 * size_mode_1_idx * sizeof(double));
    
    
    // // parallelising 'j_ptr' for contraction = 2 :
    // cudaMemset(buffer_for_contraction_2, 0, n * f2 * size_mode_1_idx * sizeof(double));
    // cudaMemset(k_buffer_for_contraction_2, 0, n * size_mode_1_idx * sizeof(int));
    
    // Stream setup
    uint64_t i, itr, j_ptr_offset;
    // uint64_t NUM_STREAMS = size_mode_0_idx;
    uint64_t NUM_STREAMS = 4; //increasing beyond 4 doesn't improve performance
    
    cudaStream_t streams[NUM_STREAMS];
    for (itr = 0; itr < NUM_STREAMS; ++itr) {
      cudaStreamCreate(&streams[itr]);
    }
    
    // uint64_t mode_1_idx_offset, mode_2_ptr_offset, mode_2_idx_offset, mode_1_idx_num_elements;
    // Launch kernels
    if (contraction == 0 || contraction == 1) {
      cout << "No. of streams = " << NUM_STREAMS <<endl;
      for (uint64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
        i = mode_0_idx[i_ptr];
        j_ptr_offset = mode_1_ptr[i_ptr];
        
        // int blocksPerGrid = mode_1_ptr[i_ptr + 1] - mode_1_ptr[i_ptr];
        dim3 gridDim(mode_1_ptr[i_ptr + 1] - mode_1_ptr[i_ptr]);
        dim3 blockDim(32, 32);
        int sharedMemBytes = f2 * sizeof(double);
        
        // mode_1_idx_offset = mode_1_ptr[i_ptr] ;
        // mode_1_idx_num_elements = mode_1_ptr[i_ptr + 1] - mode_1_ptr[i_ptr];
        // mode_2_ptr_offset = mode_2
        // mode_2_idx_offset;
        // cudaMemcpyAsync(d_mode_1_idx + mode_1_idx_offset, mode_1_idx + mode_1_idx_offset, sizeof(uint64_t) * mode_1_idx_num_elements, cudaMemcpyHostToDevice, streams[i_ptr%NUM_STREAMS]);
        // cudaMemcpyAsync(d_mode_2_ptr + mode_2_ptr_offset, mode_2_ptr + mode_2_ptr_offset, sizeof(uint64_t) * mode_2_ptr_num_elements, cudaMemcpyHostToDevice);
        // cudaMemcpyAsync(d_mode_2_idx + mode_2_idx_offset, mode_2_idx + mode_2_idx_offset, sizeof(uint64_t) * mode_2_idx_num_elememts, cudaMemcpyHostToDevice);
        // cudaMemcpyAsync(d_values + mode_2_idx_offset, values + mode_2_idx_offset, sizeof(double) * mode_2_idx_num_elememts, cudaMemcpyHostToDevice);
        
        //TO-DO: Instead, use cudaStreamQuery to find idle streams and then assign work. will it improve performance? No I think
        GPU_4loop_streams<<<gridDim, blockDim, sharedMemBytes, streams[i_ptr%NUM_STREAMS]>>>(
          // d_mode_1_ptr, 
          d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
          d_values, d_arr_A, d_arr_B, d_arr_O, l, m, n, f1, f2, contraction,
          size_mode_0_ptr, size_mode_1_ptr, size_mode_2_ptr,
          size_mode_0_idx, size_mode_1_idx, size_mode_2_idx,
          i, j_ptr_offset
        );
        cudaGetLastError();  // Check launch err;
      }
    }
    else if(contraction == 2){
      double* buffer_for_ncm_2;
      bool* k_index_buffer;
      
      NUM_STREAMS = 1;
      cout << "No. of streams = " << NUM_STREAMS <<endl;

      cudaMalloc(&buffer_for_ncm_2, n * f2 * NUM_STREAMS * sizeof(double));
      cudaMalloc(&k_index_buffer, n * NUM_STREAMS * sizeof(bool));
      
      // cudaMalloc(&k_indices, n * NUM_STREAMS * sizeof(uint64_t));
      // cudaMalloc(&counter,  NUM_STREAMS * sizeof(uint64_t));
      
      // cudaMemset(buffer_for_ncm_2 , 0, n * f2  * NUM_STREAMS * sizeof(double));
      // cudaMemset(k_index_buffer, 0, n  * NUM_STREAMS * sizeof(bool));

      
      for (uint64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
        i = mode_0_idx[i_ptr];
        j_ptr_offset = mode_1_ptr[i_ptr];
        
        cudaMemset(buffer_for_ncm_2 + n * f2 * (i_ptr % NUM_STREAMS), 0, n * f2  * sizeof(double));
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

  // Sync and destroy streams
  for ( itr = 0; itr < NUM_STREAMS; ++itr) {
    cudaStreamSynchronize(streams[itr]);
    cudaStreamDestroy(streams[itr]);
  }

  
    // Copy results back to host
    cudaMemcpy(arr_O, d_arr_O, sizeof(double) * arr_O_size, cudaMemcpyDeviceToHost);
  
    // Free device memory
    // cudaFree(d_mode_0_ptr);
    // cudaFree(d_mode_0_idx);
    // cudaFree(d_mode_1_ptr);
    cudaFree(d_mode_1_idx);
    cudaFree(d_mode_2_ptr);
    cudaFree(d_mode_2_idx);
    cudaFree(d_values);
    cudaFree(d_arr_A);
    cudaFree(d_arr_B);
    cudaFree(d_arr_O);
  
    // cudaFree(buffer_for_contraction_0_1);
    // cudaFree(buffer_for_contraction_2);
    // cudaFree(k_buffer_for_contraction_2);
  }
/*End of host function for GPU 4 loop Method using STREAMS*/
/////////////////////////////////////////////////////////////////////
int main(int argc, char** argv){
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <csf_file> [rank_1] [rank_2] [ncm]" << endl;
    std::cerr << "csf_file = Path to CSF tensor file" << endl;
    std::cerr << "rank_1 = rank of 1st factor matrix (default = 30)" << endl;
    std::cerr << "rank_2 = rank of 2nd factor matrix (default = 30)" << endl;
    std::cerr << "ncm = Non Contracting Mode (default = 0)" << endl;
    return 1;
  }

  int default_rank = 30;
  string csf_file = argv[1];
  
  // Read the CSF tensor from file
  CSFTensor tensor = readCSFTensor(csf_file);
  
  // Input tensor dimensions from the CSF file
  if (tensor.dimensions.empty() || tensor.dimensions.size() != 3) {
    cerr << "Error: CSF file must contain 3-dimensional tensor with dimensions information" << endl;
    return 1;
  }
  
  // Get dimensions from CSF file
  uint64_t dim_0 = tensor.dimensions[0];
  uint64_t dim_1 = tensor.dimensions[1];
  uint64_t dim_2 = tensor.dimensions[2];
  
  // Get command-line arguments
  uint64_t r1 = (argc > 2) ? atoi(argv[2]) : default_rank;
  uint64_t r2 = (argc > 3) ? atoi(argv[3]) : default_rank;
  int ncm = (argc > 4) ? atoi(argv[4]) : 0;

  if(ncm < 0 || ncm > 2){
    std::cerr << "Error: Contraction value must be 0, 1, or 2.\n";
    return 1;
  }
  // std::cout <<"The Tensor is of dimension: " << dim_0 << "x" << dim_1 << "x" << dim_2 << endl;
  std::cout << "The column dimensions of output factor matrices  (r1 and r2) will be : " << r1 << " and " << r2 << endl;
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

  // Input tensor dimensions (l * m * n)
  uint64_t l = dim_0;
  uint64_t m = dim_1;
  uint64_t n = dim_2;

  // Get pointers and indices for all modes
  uint64_t* mode_0_ptr = nullptr;
  uint64_t* mode_0_idx = nullptr;
  uint64_t* mode_1_ptr = nullptr;
  uint64_t* mode_1_idx = nullptr;
  uint64_t* mode_2_ptr = nullptr;
  uint64_t* mode_2_idx = nullptr;
  double* values = nullptr;
  int order = 0;

  // Convert CSFTensor to raw arrays
  getCSFArrays(tensor, &mode_0_ptr, &mode_0_idx, &mode_1_ptr, &mode_1_idx, &mode_2_ptr, &mode_2_idx, &values, &order);

  // Check if the order matches our expectation
  if (order != 3) {
    std::cerr << "Error: Expected a 3rd order tensor, but got order " << order << endl;
    return 1;
  }
  
  int size_mode_0_ptr = tensor.ptrs[0].size();
  int size_mode_0_idx = tensor.idxs[0].size();
  int size_mode_1_ptr = tensor.ptrs[1].size();
  int size_mode_1_idx = tensor.idxs[1].size();
  int size_mode_2_ptr = tensor.ptrs[2].size();
  int size_mode_2_idx = tensor.idxs[2].size();
  uint64_t total_values = tensor.values.size();

  cout << "Size of Mode 0 Pointer : " << size_mode_0_ptr << endl; 
  cout << "Size of Mode 1 Pointer : " << size_mode_1_ptr << endl; 
  cout << "Size of Mode 2 Pointer : " << size_mode_2_ptr << endl; 
  cout << "Size of Mode 0 Indices : " << size_mode_0_idx << endl; 
  cout << "Size of Mode 1 Indices : " << size_mode_1_idx << endl; 
  cout << "Size of Mode 2 Indices : " << size_mode_2_idx << endl; 
  cout << "Total non-zero values  : " << total_values << endl;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  
  unsigned int A_seed = 1, B_seed = 2;
  
  ////////////////////////////////////////////////////////////////
  // pinned memory for streams
  size_t ptr_size_0 = sizeof(uint64_t) * size_mode_0_ptr;
  size_t idx_size_0 = sizeof(uint64_t) * size_mode_0_idx;
  size_t ptr_size_1 = sizeof(uint64_t) * size_mode_1_ptr;
  size_t idx_size_1 = sizeof(uint64_t) * size_mode_1_idx;
  size_t ptr_size_2 = sizeof(uint64_t) * size_mode_2_ptr;
  size_t idx_size_2 = sizeof(uint64_t) * size_mode_2_idx;
  size_t val_size   = sizeof(double)   * total_values;
  
  // Register host memory
  cudaHostRegister(mode_0_ptr, ptr_size_0, cudaHostRegisterDefault);
  cudaHostRegister(mode_0_idx, idx_size_0, cudaHostRegisterDefault);
  cudaHostRegister(mode_1_ptr, ptr_size_1, cudaHostRegisterDefault);
  cudaHostRegister(mode_1_idx, idx_size_1, cudaHostRegisterDefault);
  cudaHostRegister(mode_2_ptr, ptr_size_2, cudaHostRegisterDefault);
  cudaHostRegister(mode_2_idx, idx_size_2, cudaHostRegisterDefault);
  cudaHostRegister(values,     val_size,   cudaHostRegisterDefault);
  ////////////////////////////////////////////////////////////////

  double* arr_A = nullptr;
  double* arr_B = nullptr;

  uint64_t arr_A_rows = 0;
  uint64_t arr_B_rows = 0;


  if (ncm == 0) {
    arr_A_rows = dim_1;
    arr_B_rows = dim_2;
  } else if (ncm == 1) {
    arr_A_rows = dim_0;
    arr_B_rows = dim_2;
  } else if (ncm == 2) {
    arr_A_rows = dim_0;
    arr_B_rows = dim_1;
  }

  generate_matrix(arr_A_rows, r1, A_seed, arr_A);
  generate_matrix(arr_B_rows, r2, B_seed, arr_B);

  uint64_t arr_A_size = arr_A_rows * r1;
  uint64_t arr_B_size = arr_B_rows * r2;

  uint64_t output_sizes[3];
  output_sizes[0] = dim_0 * r1 * r2;
  output_sizes[1] = r1 * dim_1 * r2;
  output_sizes[2] = r1 * r2 * dim_2;

  uint64_t arr_O_size = output_sizes[ncm];
  // /*
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // double* arr_O_1 = allocate_aligned_array(arr_O_size); 

  // auto start_1 = high_resolution_clock::now();
  
  // // Performing TTMC contraction using CPU - 5 for loops
  // performContraction_cpu_1(mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, mode_2_ptr, mode_2_idx, 
  //                   values, arr_A, arr_B, arr_O_1, arr_A_size, arr_B_size, arr_O_size, ncm, l, m, n, r1, r2);

  // auto end_1 = high_resolution_clock::now();
  // auto duration_1 = duration_cast<microseconds>(end_1 - start_1);
  // double seconds_1 = duration_1.count() /1e3;

  // // Output time taken with 2 decimal places
  // cout << fixed << setprecision(2); // Set fixed-point notation and precision
  // cout << "Time taken by CPU Method - 1 [5-for loop] i.e. contraction 1 : " << seconds_1 << " milliseconds" << endl;

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
  double seconds_2 = duration_2.count() /1e3;
  
  // Output time taken with 2 decimal places
  cout << fixed << setprecision(2); // Set fixed-point notation and precision
  cout << "Time taken by CPU Method - 2 [4-for loop] i.e. contraction 2 : " << seconds_2 << " milliseconds" << endl;
  
  // bool correct_cpu_1_cpu_2 = compare_matrices(arr_O_1, arr_O_2, 1, arr_O_size);
  
  // if (correct_cpu_1_cpu_2) {
  //   std::cout << "Output tensors from CPU Method-1[5-for loops] and CPU Method-2[4-for loops] are same." << std::endl;
  // } else {
  //   std::cout << "Output tensors from CPU Method-1[5-for loops] and CPU Method-2[4-for loops] are not same." << std::endl;
  // }

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
  double seconds_3 = duration_3.count() /1e3;

  // Output time taken with 2 decimal places
  cout << fixed << setprecision(2); // Set fixed-point notation and precision
  cout << "Time taken by GPU Method - 1 [5-for loop] i.e. contraction 3 : " << seconds_3 << " milliseconds" << endl;

  bool correct_cpu_2_gpu_1 = compare_matrices(arr_O_2, arr_O_3, 1, arr_O_size);

  if (correct_cpu_2_gpu_1) {
      std::cout << "Output tensors from CPU Method-1[5-for loops] and GPU Method-1[5-for loops] are same." << std::endl;
  } else {
      std::cout << "Output tensors from CPU Method-1[5-for loops] and GPU Method-1[5-for loops] are not same." << std::endl;
  }
  free(arr_O_3);
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
  double seconds_4 = duration_4.count() /1e3;
  
  // Output time taken with 2 decimal places
  cout << fixed << setprecision(2); // Set fixed-point notation and precision
  cout << "Time taken by GPU Method - 2 [4-for loop] i.e. contraction 4 : " << seconds_4 << " milliseconds" << endl;
  
  bool correct_cpu_2_gpu_2 = compare_matrices(arr_O_2, arr_O_4, 1, arr_O_size);
  
  if (correct_cpu_2_gpu_2) {
    std::cout << "Output tensors from CPU Method-2[4-for loops] and GPU Method-2[4-for loops] are same." << std::endl;
  } else {
    std::cout << "Output tensors from CPU Method-2[4-for loops] and GPU Method-2[4-for loops] are not same." << std::endl;
  }
  free(arr_O_4);
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  double* arr_O_5 =  allocate_aligned_array(arr_O_size); 
  // Record start time
  auto start_5 = high_resolution_clock::now();
  
  // Performing TTMC contraction using GPU - 4 for loops
  GPU_4loop_host_func(mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, mode_2_ptr, mode_2_idx, 
    values, arr_A, arr_B, arr_O_5, arr_A_size, arr_B_size, arr_O_size, ncm, l, m, n, r1, r2, total_values,
    size_mode_0_ptr, size_mode_1_ptr, size_mode_2_ptr, size_mode_0_idx, size_mode_1_idx, size_mode_2_idx);
    
  // Record end time
  auto end_5 = high_resolution_clock::now();
  auto duration_5 = duration_cast<microseconds>(end_5 - start_5);
  double seconds_5 = duration_5.count() /1e3;
  
  // Output time taken with 2 decimal places
  cout << fixed << setprecision(2); // Set fixed-point notation and precision
  cout << "Time taken by GPU Method - 3 [4-for loop] i.e. streams: " << seconds_5 << " milliseconds" << endl;
  
  bool correct_cpu_2_gpu_3 = compare_matrices(arr_O_2, arr_O_5, 1, arr_O_size);
  
  if (correct_cpu_2_gpu_3) {
    std::cout << "Output tensors from CPU Method-2[4-for loops] and GPU Method-3[4-for loops using streams] are same." << std::endl;
  } else {
    std::cout << "Output tensors from CPU Method-2[4-for loops] and GPU Method-3[4-for loops using streams] are not same." << std::endl;
  }
  free(arr_O_5);
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  cudaHostUnregister(mode_0_ptr);
  cudaHostUnregister(mode_0_idx);
  cudaHostUnregister(mode_1_ptr);
  cudaHostUnregister(mode_1_idx);
  cudaHostUnregister(mode_2_ptr);
  cudaHostUnregister(mode_2_idx);
  cudaHostUnregister(values);

  // Free allocated memory
  delete[] mode_0_ptr;
  delete[] mode_0_idx;
  delete[] mode_1_ptr;
  delete[] mode_1_idx;
  delete[] mode_2_ptr;
  delete[] mode_2_idx;
  delete[] values;

  std::free(arr_A);
  std::free(arr_B);
  std::free(arr_O_2);
  // std::free(arr_O_3);
  // std::free(arr_O_4);
  // std::free(arr_O_5);
}