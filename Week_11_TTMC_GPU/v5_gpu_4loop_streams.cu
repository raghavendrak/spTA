#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>
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
/*Start of device function for GPU 4 loop Method using STREAMS*/
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
  
  // parallelize s across warps
  // block dimesion is 32 x 32. 
  // hence, each row of thread block will form a warp 
  // each row of thread block(a warp) picks a k, thus a nonzero of input tensor
  for(k_ptr_offset = mode_2_ptr[j_ptr]; k_ptr_offset < mode_2_ptr[j_ptr + 1]; k_ptr_offset += blockDim.x){
    k_ptr =  k_ptr_offset + threadIdx.x;
    if(k_ptr < mode_2_ptr[j_ptr + 1]){
      
      value = values[k_ptr];
      k = mode_2_idx[k_ptr];
      
      //Each thread in a warp picks a 's'
      for(s_offset = 0; s_offset < f2; s_offset += blockDim.y){ 
        s = s_offset + threadIdx.y;
        if(s < f2){
          mask = __activemask();
          index_B = k * f2 + s;
          double prod_val = value * arr_B[index_B];

          for(int shuffle_offset = WARP_SIZE/2; shuffle_offset > 0; shuffle_offset>>=1){
            prod_val += __shfl_down_sync(mask, prod_val, shuffle_offset);
          }
          if(threadIdx.x == 0) buf[s] += prod_val;
        //   atomicAdd(&buf[s], value * arr_B[index_B] );
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
        double *values;
        int order;
        
        getCSFArrays(tensor, &mode_0_ptr, &mode_0_idx, 
                    &mode_1_ptr, &mode_1_idx, 
                    &mode_2_ptr, &mode_2_idx, 
                    &values, &order);
        
        size_t size_mode_0_ptr = tensor.ptrs[0].size();
        size_t size_mode_1_ptr = tensor.ptrs[1].size();
        size_t size_mode_2_ptr = tensor.ptrs[2].size();
        size_t size_mode_0_idx = tensor.idxs[0].size();
        size_t size_mode_1_idx = tensor.idxs[1].size();
        size_t size_mode_2_idx = tensor.idxs[2].size();
        size_t total_values = tensor.values.size();
        
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
        
        // Run this implementation (GPU 4-loop with streams) first
        if (verbose) {
            cout << "Running GPU 4-loop with streams implementation..." << endl;
        }
        auto start = std::chrono::high_resolution_clock::now();
        
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

        GPU_4loop_host_func(
            mode_0_ptr, mode_0_idx,
            mode_1_ptr, mode_1_idx,
            mode_2_ptr, mode_2_idx,
            values, arr_A, arr_B, arr_O,
            arr_A_size, arr_B_size, arr_O_size,
            ncm, tensor.dimensions[0], tensor.dimensions[1], tensor.dimensions[2], rank1, rank2,
            tensor.values.size(),
            size_mode_0_ptr, size_mode_1_ptr, size_mode_2_ptr,
            size_mode_0_idx, size_mode_1_idx, size_mode_2_idx
        );
        
        cudaHostUnregister(mode_0_ptr);
        cudaHostUnregister(mode_0_idx);
        cudaHostUnregister(mode_1_ptr);
        cudaHostUnregister(mode_1_idx);
        cudaHostUnregister(mode_2_ptr);
        cudaHostUnregister(mode_2_idx);
        cudaHostUnregister(values);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
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
            tensor.dimensions[0], tensor.dimensions[1], tensor.dimensions[2], rank1, rank2
        );
        
        auto ref_end = std::chrono::high_resolution_clock::now();
        auto ref_duration = std::chrono::duration_cast<std::chrono::microseconds>(ref_end - ref_start).count();
        
        // Validate results using compare_results from matrix_utils.h
        bool valid = compare_results(arr_O, ref_O, arr_O_size);
        
        // Report results
        if (verbose) {
            cout << "GPU 4-loop streams execution time: " << duration / 1000.0 << " ms" << endl;
            cout << "Reference execution time: " << ref_duration / 1000.0 << " ms" << endl;
            cout << "Speedup over reference: " << (double)ref_duration / duration << "x" << endl;
            cout << "Result validation: " << (valid ? "PASSED" : "FAILED") << endl;
        } else {
            cout << "Method: GPU 4-loop streams, Time: " << duration / 1000.0 << " ms, Validation: " << (valid ? "PASSED" : "FAILED") << endl;
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