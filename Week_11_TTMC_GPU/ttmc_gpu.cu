#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <cstring>  
#include <iomanip> 
#include <cuda_runtime.h>
#include "genten.h"      // Include the header for genten
#include "COO_to_CSF.h"

using namespace std;
using namespace std::chrono;


//////////////////////////////////////////////////////////
/* Start of Function for reading a matrix from .txt file*/

// Function to read a matrix from a file 
void readMatrix(const string& filename, int64_t& rows, int64_t& cols, double*& arr) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Unable to open matrix file: " + filename);
    }

    // Read the entire file into a vector
    arr = new double[rows * cols];
    double value;
    
    int64_t count = 0;
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

void writeMatrixToFile(const std::string& filename, int64_t rows, int64_t cols, double val) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            file << val << " ";
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




///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
/* Start of Functions of GPU Method - 1 [5-for loops]*/

__global__ void contractionKernel_0(
    int64_t* mode_0_ptr, int64_t* mode_0_idx,
    int64_t* mode_1_ptr, int64_t* mode_1_idx,
    int64_t* mode_2_ptr, int64_t* mode_2_idx,
    double* values, double* arr_A, double* arr_B,  
    double* arr_O, int64_t l, int64_t m, int64_t n, int64_t f1, int64_t f2, int contraction) 
{
    // Compute thread indices
    int64_t i_ptr = blockIdx.x * blockDim.x + threadIdx.x;

    if (i_ptr < mode_0_ptr[1]) {
        int64_t i = mode_0_idx[i_ptr] - 1;

        for (int64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
            int64_t j = mode_1_idx[j_ptr] - 1;

            for (int64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
                int64_t k = mode_2_idx[k_ptr] - 1;
                double value = values[k_ptr];

                for (int64_t r = 0; r < f1; ++r) {
                    int64_t index_A = 0;
                    if(contraction == 0){
                        index_A = j * f1 + r;
                    }
                    else if(contraction == 1){
                        index_A = i * f1 + r;
                    }
                    else if(contraction == 2){
                        index_A = i * f1 + r;
                    }
                    for (int64_t s = 0; s < f2; ++s) {
                        int64_t index_B = 0;
                        int64_t index_O = 0;
                        if(contraction == 0){
                            index_B = k * f2 + s;
                            index_O = s * l * f1 + i * f1 + r;
                        }
                        else if(contraction == 1){
                            index_B = k * f2 + s;
                            index_O = s * m * f1 + r * m + j;
                        }
                        else if(contraction == 2){
                            index_B = j * f2 + s;
                            index_O = k * f1 * f2 + r * f2 + s;
                        }

                        atomicAdd_double(&arr_O[index_O], value * arr_A[index_A] * arr_B[index_B]);
                    }
                }
            }
        }
    }
}


void performContraction_gpu_1(
    int64_t* mode_0_ptr, int64_t* mode_0_idx,
    int64_t* mode_1_ptr, int64_t* mode_1_idx,
    int64_t* mode_2_ptr, int64_t* mode_2_idx,
    double* values, double* arr_A, double* arr_B,  
    double* arr_O, int64_t arr_A_size, int64_t arr_B_size, int64_t arr_O_size, int contraction, 
    int64_t l, int64_t m, int64_t n, int64_t f1, int64_t f2, int64_t total_values, int size_mode_0_ptr, int size_mode_1_ptr, int size_mode_2_ptr, int size_mode_0_idx, int size_mode_1_idx, int size_mode_2_idx) 
{
    // Allocate device memory
    int64_t *d_mode_0_ptr, *d_mode_0_idx, *d_mode_1_ptr, *d_mode_1_idx, *d_mode_2_ptr, *d_mode_2_idx;
    double *d_values, *d_arr_A, *d_arr_B, *d_arr_O;

    cudaMalloc(&d_mode_0_ptr, sizeof(int64_t) * size_mode_0_ptr);
    cudaMalloc(&d_mode_0_idx, sizeof(int64_t) * size_mode_0_idx);
    cudaMalloc(&d_mode_1_ptr, sizeof(int64_t) * size_mode_1_ptr);
    cudaMalloc(&d_mode_1_idx, sizeof(int64_t) * size_mode_1_idx);
    cudaMalloc(&d_mode_2_ptr, sizeof(int64_t) * size_mode_2_ptr);
    cudaMalloc(&d_mode_2_idx, sizeof(int64_t) * size_mode_2_idx);
    cudaMalloc(&d_values, sizeof(double) * total_values);
    cudaMalloc(&d_arr_A, sizeof(double) * arr_A_size);
    cudaMalloc(&d_arr_B, sizeof(double) * arr_B_size);
    cudaMalloc(&d_arr_O, sizeof(double) * arr_O_size);

    // Copy data from host to device
    cudaMemcpy(d_mode_0_ptr, mode_0_ptr, sizeof(int64_t) * size_mode_0_ptr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_0_idx, mode_0_idx, sizeof(int64_t) * size_mode_0_idx, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_1_ptr, mode_1_ptr, sizeof(int64_t) * size_mode_1_ptr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_1_idx, mode_1_idx, sizeof(int64_t) * size_mode_1_idx, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_2_ptr, mode_2_ptr, sizeof(int64_t) * size_mode_2_ptr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode_2_idx, mode_2_idx, sizeof(int64_t) * size_mode_2_idx, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, sizeof(double) * total_values, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_A, arr_A, sizeof(double) * arr_A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_B, arr_B, sizeof(double) * arr_B_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_O, arr_O, sizeof(double) * arr_O_size, cudaMemcpyHostToDevice);

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (size_mode_0_idx + threadsPerBlock - 1) / threadsPerBlock;

    // Launch appropriate kernel based on contraction type
    contractionKernel_0<<<blocksPerGrid, threadsPerBlock>>>(
        d_mode_0_ptr, d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
        d_values, d_arr_A, d_arr_B, d_arr_O, l, m, n, f1, f2, contraction);

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

/* End of Functions of GPU Method - 1 [5-for loops]*/
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////








///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
/* Start of Functions of CPU Method - 1 [5-for loops]*/

// Function to perform contraction based on the inputs using 5 for loops
void performContraction_cpu_1(int64_t*& mode_0_ptr, int64_t*& mode_0_idx,
                        int64_t*& mode_1_ptr, int64_t*& mode_1_idx,
                        int64_t*& mode_2_ptr, int64_t*& mode_2_idx,
                        double*& values, double*& arr_A, double*& arr_B,  
                        double*& arr_O, int64_t& arr_A_size, int64_t& arr_B_size, int64_t& arr_O_size, int& contraction, 
                        int64_t& l, int64_t& m, int64_t& n, int64_t& f1, int64_t& f2) {

    if(contraction == 0){
        // Traverse through CSF tensor pointer and indices arrays for all modes
        for (int64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
            int64_t i = mode_0_idx[i_ptr] - 1;                         // Index in the mode 'i'

            for (int64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
                int64_t j = mode_1_idx[j_ptr] - 1;                     // Index for 'j' mode in CSF

                for (int64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
                    int64_t k = mode_2_idx[k_ptr] - 1;                 // Index for 'k' mode in CSF

                    double value = values[k_ptr];                  // CSF value for the above i, j, k

                    // Iterate over the matrix dimensions 
                    for (int64_t r = 0; r < f1; ++r) {
                        
                        int64_t index_A = j * f1 + r;
                        for (int64_t s = 0; s < f2; ++s) {

                            // Compute linearized indices for matrices A, B based on the contraction string
                            int64_t index_B = k * f2 + s;

                            // For mode-1 linearized output 
                            int64_t index_O = s * l * f1 + i * f1 + r;

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
        for (int64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
            int64_t i = mode_0_idx[i_ptr] - 1;                         // Index in the mode 'i'

            for (int64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
                int64_t j = mode_1_idx[j_ptr] - 1;                     // Index for 'j' mode in CSF

                for (int64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
                    int64_t k = mode_2_idx[k_ptr] - 1;                 // Index for 'k' mode in CSF

                    double value = values[k_ptr];                  // CSF value for the above i, j, k

                    // Iterate over the matrix dimensions 
                    for (int64_t r = 0; r < f1; ++r) {
                        
                        int64_t index_A = i * f1 + r;
                        for (int64_t s = 0; s < f2; ++s) {

                            // Compute linearized indices for matrices A, B based on the contraction string
                            int64_t index_B = k * f2 + s;

                            // For mode-1 linearized output 
                            int64_t index_O = s * m * f1 + r * m + j;

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
        for (int64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
            int64_t i = mode_0_idx[i_ptr] - 1;                         // Index in the mode 'i'

            for (int64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
                int64_t j = mode_1_idx[j_ptr] - 1;                     // Index for 'j' mode in CSF

                for (int64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
                    int64_t k = mode_2_idx[k_ptr] - 1;                 // Index for 'k' mode in CSF

                    double value = values[k_ptr];                  // CSF value for the above i, j, k

                    // Iterate over the matrix dimensions 
                    for (int64_t r = 0; r < f1; ++r) {
                        
                        int64_t index_A = i * f1 + r;
                        for (int64_t s = 0; s < f2; ++s) {

                            // Compute linearized indices for matrices A, B based on the contraction string
                            int64_t index_B = j * f2 + s;

                            // For mode-1 linearized output
                            int64_t index_O = k * f1 * f2 + r * f2 + s;

                            // Perform contraction
                            arr_O[index_O] += value * arr_A[index_A] * arr_B[index_B];            
                        }
                    }
                }
            }
        }
    }
}


/* End of Functions of CPU Method - 1 [5-for loops]*/
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////


///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
/* Start of Functions of CPU Method - 2 [4-for loops]*/

// Function to perform contraction based on the inputs using 4 for loops
void performContraction_cpu_2(int64_t*& mode_0_ptr,int64_t*& mode_0_idx,
                        int64_t*& mode_1_ptr, int64_t*& mode_1_idx,
                        int64_t*& mode_2_ptr, int64_t*& mode_2_idx,
                        double*& values, double*& arr_A, double*& arr_B,  
                        double*& arr_O, int64_t& arr_A_size, int64_t& arr_B_size, int64_t& arr_O_size, int& contraction, 
                        int64_t& l, int64_t& m, int64_t& n, int64_t& f1, int64_t& f2) {
    
                            
    if(contraction == 0){
        double* buffer = new double[f2];    // buffer for mode-s

        // Traverse through CSF tensor pointer and indices arrays for all modes
        for (int64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
            int64_t i = mode_0_idx[i_ptr] - 1;                         // Index in the mode 'i'
            
            
            for (int64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
                int64_t j = mode_1_idx[j_ptr] - 1;                     // Index for 'j' mode in CSF
                
                memset(buffer, 0, f2 * sizeof(double));             // Set the entire memory block to 0
                
                for (int64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
                    int64_t k = mode_2_idx[k_ptr] - 1;                 // Index for 'k' mode in CSF

                    double value = values[k_ptr];                  // CSF value for the above i, j, k

                    for (int64_t s = 0; s < f2; ++s) {

                        // Compute linearized indices for matrices B based on the contraction string
                        int64_t index_B = k * f2 + s;

                        buffer[s] += value * arr_B[index_B];                        
                    }
                }

                for (int64_t r = 0; r < f1; ++r) {
                    
                    // Compute linearized indices for matrices B based on the contraction string
                    int64_t index_A = j * f1 + r;
                    for (int64_t s = 0; s < f2; ++s) {

                        // For mode-1 linearized output
                        int64_t index_O = s * l * f1 + i * f1 + r;

                        // Perform contraction
                        arr_O[index_O] += buffer[s] * arr_A[index_A];              
                    }
                }
            }
        }

        delete [] buffer;
    }
    else if(contraction == 1){
        double* buffer = new double[f2];    // buffer for mode-s
        
        // Traverse through CSF tensor pointer and indices arrays for all modes
        for (int64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
            int64_t i = mode_0_idx[i_ptr] - 1;                         // Index in the mode 'i'

            for (int64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
                int64_t j = mode_1_idx[j_ptr] - 1;                     // Index for 'j' mode in CSF

                memset(buffer, 0, f2 * sizeof(double));             // Set the entire memory block to 0

                for (int64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
                    int64_t k = mode_2_idx[k_ptr] - 1;                 // Index for 'k' mode in CSF

                    double value = values[k_ptr];                  // CSF value for the above i, j, k

                    for (int64_t s = 0; s < f2; ++s) {

                        // Compute linearized indices for matrices B based on the contraction string
                        int64_t index_B = k * f2 + s;

                        // Perform contraction
                        buffer[s] += value * arr_B[index_B];            
                    }
                }


                for (int64_t r = 0; r < f1; ++r) {

                    // Compute linearized indices for matrices A, B based on the contraction string
                    int64_t index_A = i * f1 + r;
                    for (int64_t s = 0; s < f2; ++s) {

                        // For mode-1 linearized output 
                        int64_t index_O = s * m * f1 + r * m + j;

                        // Perform contraction
                        arr_O[index_O] += buffer[s] * arr_A[index_A];              
                    }
                }
            }
        }

        delete [] buffer;
    }
    else if(contraction == 2){
        double* buffer = new double[n*f2];    // buffer for mode-k and mode-s
        int64_t* k_buffer = new int64_t[n];  // buffer for k-indices

        // Traverse through CSF tensor pointer and indices arrays for all modes
        for (int64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
            int64_t i = mode_0_idx[i_ptr] - 1;                          // Index in the mode 'i'

            memset(buffer, 0, n * f2 * sizeof(double));             // Set the entire memory block to 0
            memset(k_buffer, 0, n * sizeof(int64_t));
            for (int64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
                int64_t j = mode_1_idx[j_ptr] - 1;                      // Index for 'j' mode in CSF

                for (int64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
                    int64_t k = mode_2_idx[k_ptr] - 1;                  // Index for 'k' mode in CSF
                    k_buffer[k]++;

                    double value = values[k_ptr];                   // CSF value for the above i, j, k

                    for (int64_t s = 0; s < f2; ++s) {

                        // Compute linearized indices for matrices B based on the contraction string
                        int64_t index_B = j * f2 + s;

                        int64_t index_buf = k * f2 + s; 

                        buffer[index_buf] += value * arr_B[index_B];          
                    }
                }
            }


            for (int64_t z = 0; z < n ; ++z) {
                int64_t k = z;

                if(k_buffer[z] > 0){
                    for (int64_t r = 0; r < f1; ++r) {

                        // Compute linearized indices for matrices A based on the contraction string
                        int64_t index_A = i * f1 + r;
                        for (int64_t s = 0; s < f2; ++s) {
                    
                            // For mode-1 linearized output 
                            int64_t index_O = k * f1 * f2 + r * f2 + s;

                            int64_t index_buf = k * f2 + s; 

                            arr_O[index_O] += buffer[index_buf] * arr_A[index_A] ;        
                        }
                    }
                }
            }
        }

        delete [] buffer;
        delete [] k_buffer;
    }


    
}

/* End of Functions of CPU Method - 2 [4-for loops]*/
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////


/*
NOTE : Make sure to change the input matrix files as per the chosen contraction : 

OUTPUT for the below commands :

1       3       100     500     500     1       nz_slc_cnt      100     100     density_slc     1       1       1       distr_type   2        max_fib_per_slc         500     101     0.202   imbal_fib_per_slc       -1      1.1335          -1.1335         std_fib_per_slc         25      16.0961         0.643844      cv_fib_per_slc          0.5     0.340011        0.680021        avg_fib_per_slc         50      47.34   0.9468          nz_fib_cnt      5000    4734    density_fib  0.1      0.09468         1.1562          0.9468          distr_type      2       max_nz_per_fib          500     220     0.44    imbal_nz_per_fib        -1      3.42695      -3.42695         std_nz_per_fib          25      23.2328         0.929314        cv_nz_per_fib   0.5     0.467503        0.935006        avg_nz_per_fib          50      49.6956       0.993912        nnz     250000          235259          density         0.01    0.00941036      1.00189         0.941036        48      TIME    0.0034001       0.0052953     0.0009651       0.0009128       0.0172926       0.0281677 
 Order of the Tensor : 3
Dimension - 0 : 100
Dimension - 1 : 500
Dimension - 2 : 500
Total size of my_tensor_indices : 705777
Total size of my_tensor_values : 235259
Tensor in COO Format : 
Mode 0 Pointer:
0 100 
Mode 0 Indices:
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 Enter the column dimension of input matrices A and B respectively (f1) and (f2): 2 2
Choose the contraction, "0" for ijk,jr,ks→irs, "1" for ijk,ir,ks→rjs, "2" for ijk,ir,js→rsk 
0
Your Contraction Choice : ijk,jr,ks→irs
Time taken by contraction 1 : 1019986 nanoseconds
Output written to output_1.txt
Time taken by contraction 2 : 957376 nanoseconds
Output written to output_2.txt
Multiplication results are correct.
munmap_chunk(): invalid pointer
Aborted (core dumped)

*/

// Command for compiling this program : 
// nvcc -o ttmc_gpu ttmc_gpu.cu genten.c COO_to_CSF.cpp -Xcompiler -fopenmp --extended-lambda


// argv[1] -> order
// argv[2] -> dim_0
// argv[3] -> dim_1
// argv[4] -> dim_2
// argv[5] -> Number of columns for matrix A
// argv[6] -> Number of columns for matrix B
// argv[7] -> contraction choice [ 0 -> ijk,jr,ks→irs,  1 -> ijk,ir,ks→rjs, 2 -> ijk,ir,js→rsk ]

// Command to run this program : 
// ./ttmc_gpu 3 2500 2500 2500 30 30 0 -d 0.01 -f 0.1 -c 0.5 -v 0.5 


int main(int argc, char* argv[]) {
    // Check for the correct number of arguments
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <order> <dim_0> <dim_1> <dim_2> ..." << endl;
        return 1;
    }

    // Save the first four arguments
    int order = atoi(argv[1]);
    int64_t dim_0 = atoi(argv[2]);
    int64_t dim_1 = atoi(argv[3]);
    int64_t dim_2 = atoi(argv[4]);
    int64_t f1 = atoi(argv[5]);
    int64_t f2 = atoi(argv[6]);
    int contraction = atoi(argv[7]);

    // Check if contraction is set correctly
    if (contraction < 0 || contraction > 2) {
        std::cerr << "Error: Contraction value must be 0, 1, or 2.\n";
        return 1;
    }



    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////
    /* Creating the input_matrix_A.txt and input_matrix_B.txt files for better automation */

    // Determine matrix dimensions based on contraction type
    int64_t rows_A, cols_A = f1, rows_B, cols_B = f2;

    if (contraction == 0) {
        rows_A = dim_1;
        rows_B = dim_2;
    } else if (contraction == 1) {
        rows_A = dim_0;
        rows_B = dim_2;
    } else if (contraction == 2) {
        rows_A = dim_0;
        rows_B = dim_1;
    } else {
        std::cerr << "Error: Invalid contraction type. Must be 0, 1, or 2." << std::endl;
        return 1;
    }

    double val = 1.0;

    // Write matrices to files
    writeMatrixToFile("input_matrix_A.txt", rows_A, cols_A, val);
    writeMatrixToFile("input_matrix_B.txt", rows_B, cols_B, val);

    std::cout << "Matrices written to input_matrix_A.txt and input_matrix_B.txt with dimensions:\n";
    std::cout << "Matrix A: " << rows_A << " x " << cols_A << "\n";
    std::cout << "Matrix B: " << rows_B << " x " << cols_B << "\n";

    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////

    int64_t* my_tensor_indices = nullptr;
    double* my_tensor_values = nullptr;
    int64_t total_indices = 0;
    int64_t total_values = 0;

    generate_tensor(argc, argv, &my_tensor_indices, &my_tensor_values, &total_indices, &total_values);


    cout << "\nOrder of the Tensor : " << order << endl;
    cout << "Dimension - 0 : " << dim_0 << endl;
    cout << "Dimension - 1 : " << dim_1 << endl;
    cout << "Dimension - 2 : " << dim_2 << endl;
    cout << "Total size of my_tensor_indices : " << total_indices<< endl;
    cout << "Total size of my_tensor_values : " << total_values << endl;

    // cout << "Tensor in COO Format : " << endl;
    // for(int i=0, j=0; i<total_indices; i++){
    //     cout << my_tensor_indices[i] << " ";
    //     if((i+1)%3 == 0){
    //         cout << my_tensor_values[j];
    //         j++;
    //         cout << endl;
    //     }
    // }
    // cout << endl;

    cooToCSF(my_tensor_indices, my_tensor_values, order, total_indices, total_values);


    // Input tensor dimensions (l * m * n)
    int64_t l, m, n;

    l = dim_0;
    m = dim_1;
    n = dim_2;

    int64_t* mode_0_ptr = nullptr;
    int64_t* mode_0_idx = nullptr;
    int64_t* mode_1_ptr = nullptr;
    int64_t* mode_1_idx = nullptr;
    int64_t* mode_2_ptr = nullptr;
    int64_t* mode_2_idx = nullptr;
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

    cout << "Size of Mode 0 Pointer : " << size_mode_0_ptr << endl; 
    cout << "Size of Mode 1 Pointer : " << size_mode_1_ptr << endl; 
    cout << "Size of Mode 2 Pointer : " << size_mode_2_ptr << endl; 
    cout << "Size of Mode 0 Indices : " << size_mode_0_idx << endl; 
    cout << "Size of Mode 1 Indices : " << size_mode_1_idx << endl; 
    cout << "Size of Mode 2 Indices : " << size_mode_2_idx << endl; 
    // Iterate through mode_0 pointers and indices
    // std::cout << "Mode 0 Pointer:\n";
    // for (int i = 0; i < size_mode_0_ptr; ++i) {
    //     std::cout << mode_0_ptr[i] << " ";
    // }
    // std::cout << "\nMode 0 Indices:\n";
    // for (int i = 0; i < size_mode_0_idx; ++i) {
    //     std::cout << mode_0_idx[i] << " ";
    // }

    // // Iterate through mode_1 pointers and indices
    // std::cout << "\n\nMode 1 Pointer:\n";
    // for (int i = 0; i < size_mode_1_ptr; ++i) {
    //     std::cout << mode_1_ptr[i] << " ";
    // }
    // std::cout << "\nMode 1 Indices:\n";
    // for (int i = 0; i < size_mode_1_idx; ++i) {
    //     std::cout << mode_1_idx[i] << " ";
    // }

    // // Iterate through mode_2 pointers and indices
    // std::cout << "\n\nMode 2 Pointer:\n";
    // for (int i = 0; i < size_mode_2_ptr; ++i) {
    //     std::cout << mode_2_ptr[i] << " ";
    // }
    // std::cout << "\nMode 2 Indices:\n";
    // for (int i = 0; i < size_mode_2_idx; ++i) {
    //     std::cout << mode_2_idx[i] << " ";
    // }

    // // If you want to iterate through the values as well
    // std::cout << "\n\nValues:\n";
    // for (int i = 0; i < total_values; ++i) {
    //     std::cout << values[i] << " ";
    // }


    cout << "\nThe column dimensions of input matrices A (f1) : " << f1 << endl;
    cout << "The column dimensions of input matrices A (f2) : " << f2 << endl;

    double* arr_A = nullptr;
    double* arr_B = nullptr;

    if(contraction == 0){
        cout << "Your Contraction Choice : ijk,jr,ks→irs" << endl; 
    }
    else if(contraction == 1){
        cout << "Your Contraction Choice : ijk,ir,ks→rjs" << endl;
    }
    else if(contraction == 2){
        cout << "Your Contraction Choice : ijk,ir,js→rsk" << endl;
    }

    int64_t arr_A_rows = 0;
    int64_t arr_B_rows = 0;

    if(contraction == 0) {
        arr_A = new double[m * f1];
        arr_B = new double[n * f2];
        arr_A_rows = m;
        arr_B_rows = n;
    } else if(contraction == 1) {
        arr_A = new double[l * f1];
        arr_B = new double[n * f2];
        arr_A_rows = l;
        arr_B_rows = n;
    } else if(contraction == 2) {
        arr_A = new double[l * f1];
        arr_B = new double[m * f2];
        arr_A_rows = l;
        arr_B_rows = m;
    }

    readMatrix("input_matrix_A.txt", arr_A_rows, f1, arr_A);
    readMatrix("input_matrix_B.txt", arr_B_rows, f2, arr_B);

    int64_t arr_A_size = arr_A_rows * f1;
    int64_t arr_B_size = arr_B_rows * f2;

    int64_t output_sizes[3];
    output_sizes[0] = l * f1 * f2;
    output_sizes[1] = f1 * m * f2;
    output_sizes[2] = f1 * f2 * n;

    int64_t arr_O_size = output_sizes[contraction];

    double* arr_O_1 = new double[arr_O_size](); // The () initializes the array to zero

    // cout << "arr_0size : " << arr_O_size << endl;
    // for(int i=0; i<arr_O_size; i++){
    //     cout << arr_O[i] << " ";
    // }
    // cout << endl;

    // Record start time
    auto start_1 = high_resolution_clock::now();

    // Performing TTMC contraction using 5 for loops
    performContraction_cpu_1(mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, mode_2_ptr, mode_2_idx, 
                      values, arr_A, arr_B, arr_O_1, arr_A_size, arr_B_size, arr_O_size, contraction, l, m, n, f1, f2);

    // Record end time
    auto end_1 = high_resolution_clock::now();
    auto duration_1 = duration_cast<microseconds>(end_1 - start_1);
    // cout << "Time taken by contraction 1 : " << duration_1.count() << " microseconds" << endl;
    double seconds_1 = duration_1.count() / 1e6; // Convert microseconds to seconds

    // Output time taken with 2 decimal places
    cout << fixed << setprecision(2); // Set fixed-point notation and precision
    cout << "Time taken by CPU Method - 1 [5-for loop] i.e. contraction 1 : " << seconds_1 << " seconds" << endl;

    // Write the output array to output.txt
    // ofstream output_file_1("output_1.txt");
    // if (!output_file_1.is_open()) {
    //     throw runtime_error("Unable to open output file.");
    // }

    // for (int64_t i = 0; i < arr_O_size; i++) {
    //     output_file_1 << arr_O_1[i] << " ";
    // }
    // output_file_1.close();

    // cout << "Output written to output_1.txt" << endl;





    // Performing TTMC contraction using 4 for loops
    double* arr_O_2 = new double[arr_O_size](); // The () initializes the array to zero

    // Record start time
    auto start_2 = high_resolution_clock::now();

    // Perform contraction using 4 for loops
    performContraction_cpu_2(mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, mode_2_ptr, mode_2_idx, 
                       values, arr_A, arr_B, arr_O_2, arr_A_size, arr_B_size, arr_O_size, contraction, l, m, n, f1, f2);

    // Record end time
    auto end_2 = high_resolution_clock::now();
    auto duration_2 = duration_cast<microseconds>(end_2 - start_2);
    // cout << "Time taken by contraction 2 : " << duration_2.count() << " microseconds" << endl;
    double seconds_2 = duration_2.count() / 1e6; // Convert microseconds to seconds

    // Output time taken with 2 decimal places
    cout << fixed << setprecision(2); // Set fixed-point notation and precision
    cout << "Time taken by CPU Method - 2 [4-for loop] i.e. contraction 2 : " << seconds_2 << " seconds" << endl;

    // Write the output array to output.txt
    // ofstream output_file_2("output_2.txt");
    // if (!output_file_2.is_open()) {
    //     throw runtime_error("Unable to open output file.");
    // }

    // for (int64_t i = 0; i < arr_O_size; i++) {
    //     output_file_2 << arr_O_2[i] << " ";
    // }
    // output_file_2.close();

    // cout << "Output written to output_2.txt" << endl;


    bool correct_cpu_1_cpu_2 = compare_matrices(arr_O_1, arr_O_2, 1, arr_O_size);

    if (correct_cpu_1_cpu_2) {
        std::cout << "Output tensors from CPU Method-1[5-for loops] and CPU Method-2[4-for loops] are same." << std::endl;
    } else {
        std::cout << "Output tensors from CPU Method-1[5-for loops] and CPU Method-2[4-for loops] are same." << std::endl;
    }


    double* arr_O_3 = new double[arr_O_size](); // The () initializes the array to zero

    // cout << "arr_0size : " << arr_O_size << endl;
    // for(int i=0; i<arr_O_size; i++){
    //     cout << arr_O[i] << " ";
    // }
    // cout << endl;

    // Record start time
    auto start_3 = high_resolution_clock::now();

    // Performing TTMC contraction using 5 for loops
    performContraction_gpu_1(mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, mode_2_ptr, mode_2_idx, 
                       values, arr_A, arr_B, arr_O_3, arr_A_size, arr_B_size, arr_O_size, contraction, l, m, n, f1, f2, total_values,
                       size_mode_0_ptr, size_mode_1_ptr, size_mode_2_ptr, size_mode_0_idx, size_mode_1_idx, size_mode_2_idx);

    // Record end time
    auto end_3 = high_resolution_clock::now();
    auto duration_3 = duration_cast<microseconds>(end_3 - start_3);
    // cout << "Time taken by contraction 1 : " << duration_1.count() << " microseconds" << endl;
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




    double* arr_O_4 = new double[arr_O_size](); // The () initializes the array to zero

    // cout << "arr_0size : " << arr_O_size << endl;
    // for(int i=0; i<arr_O_size; i++){
    //     cout << arr_O[i] << " ";
    // }
    // cout << endl;

    // Record start time
    auto start_4 = high_resolution_clock::now();

    // Performing TTMC contraction using 5 for loops
    //performContraction_gpu_2(mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, mode_2_ptr, mode_2_idx, 
    //                   values, arr_A, arr_B, arr_O_4, arr_A_size, arr_B_size, arr_O_size, contraction, l, m, n, f1, f2, total_values,
    //                   size_mode_0_ptr, size_mode_1_ptr, size_mode_2_ptr, size_mode_0_idx, size_mode_1_idx, size_mode_2_idx);

    // Record end time
    auto end_4 = high_resolution_clock::now();
    auto duration_4 = duration_cast<microseconds>(end_4 - start_4);
    // cout << "Time taken by contraction 1 : " << duration_1.count() << " microseconds" << endl;
    double seconds_4 = duration_4.count() / 1e6; // Convert microseconds to seconds

    // Output time taken with 2 decimal places
    cout << fixed << setprecision(2); // Set fixed-point notation and precision
    cout << "Time taken by GPU Method - 2 [4-for loop] i.e. contraction 4 : " << seconds_4 << " seconds" << endl;

    bool correct_cpu_2_gpu_2 = compare_matrices(arr_O_2, arr_O_4, 1, arr_O_size);

    if (correct_cpu_2_gpu_2) {
        std::cout << "Output tensors from CPU Method-2[4-for loops] and GPU Method-2[4-for loops] are same." << std::endl;
    } else {
        std::cout << "Output tensors from CPU Method-2[4-for loops] and GPU Method-2[4-for loops] are not same." << std::endl;
    }

    

    delete[] arr_O_1;
    delete[] arr_O_2;
    delete[] arr_O_3;
    delete[] arr_O_4;



    // Memory clean up
    delete[] mode_0_ptr;
    delete[] mode_0_idx;
    delete[] mode_1_ptr;
    delete[] mode_1_idx;
    delete[] mode_2_ptr;
    delete[] mode_2_idx;
    delete[] arr_A;
    delete[] arr_B;

    delete[] my_tensor_indices; 
    delete[] my_tensor_values; 

    return 0;
}

