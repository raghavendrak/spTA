#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <cuda_runtime.h>
#include <fstream>
#include <tuple>
#include <algorithm>
#include <sstream>
#include <cstdint>  
#include "genten.h"
#include <chrono>
#include <string>
#include <cstdint>

using namespace std;

#define MAX_ORDER 8  // Assuming a maximum tensor order
#define BLOCK_SIZE 256


struct COOElement {
    int64_t indices[MAX_ORDER];  // Store indices for all modes
    double value;
};

int64_t *fiberStartFlags;
COOElement *sortedCOO;


__device__ inline int64_t atomicAddInt64(int64_t* address, int64_t val) {
    return atomicAdd((unsigned long long*)address, (unsigned long long)val);
}


// Kernel to detect fibers and populate fiber start flags
__global__ void detect_fibers(const COOElement* cooData, int64_t nnz, int order, int64_t* fiberStartFlags, int64_t* counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;

    if (tid == 0) {
        for (int level = 0; level < order; ++level) {
            fiberStartFlags[order * tid + level] = 1;  // First element is always a fiber start
            atomicAddInt64(&counts[level], 1);
        }
    } else {

        for (int level = 0; level < order; ++level) {
            if (cooData[tid].indices[level] != cooData[tid - 1].indices[level]) {
                while(level < order){
                    fiberStartFlags[order * tid + level] = 1;
                    atomicAddInt64(&counts[level], 1);
                    level++;
                }
            }
        }
    }
}


// __global__ void inclusive_scan_nonzero(int64_t* input, int64_t* output, int64_t n) {
//     // Shared memory for block-level prefix sum
//     __shared__ int64_t sh_data[BLOCK_SIZE];
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int local_tid = threadIdx.x;

//     // Load data into shared memory
//     sh_data[local_tid] = (tid < n) ? input[tid] : 0;
//     __syncthreads();

//     // Compute prefix sum (inclusive scan) for non-zero values
//     for (int stride = 1; stride < blockDim.x; stride *= 2) {
//         int temp = 0;
//         if (local_tid >= stride && sh_data[local_tid] != 0) {
//             temp = sh_data[local_tid - stride];
//         }
//         __syncthreads(); // Wait for all threads to update
//         sh_data[local_tid] += temp; // Add prefix sum
//         __syncthreads(); // Synchronize after update
//     }

//     // Write results to output array
//     if (tid < n) {
//         output[tid] = (input[tid] == 0) ? 0 : sh_data[local_tid];
//     }
// }



// void computePrefixSumNonZero(int64_t* d_input, int64_t* d_output, int64_t n) {
//     int blockSize = 256;  // Threads per block
//     int numBlocks = (n + blockSize - 1) / blockSize;  // Calculate the number of blocks

//     // Launch the kernel
//     inclusive_scan_nonzero<<<numBlocks, blockSize>>>(d_input, d_output, n);

//     // Synchronize to ensure the kernel is complete
//     cudaDeviceSynchronize();
// }


// void computePrefixSumPerOrder(int64_t* d_fiberStartFlags, int order, int64_t nnz,
//                               int64_t* d_prefixSumResults) {
//     for (int level = 0; level < order; ++level) {
//         // Extract the column
//         int64_t* d_columnFlags;
//         cudaMalloc(&d_columnFlags, nnz * sizeof(int64_t));
//         cudaMemcpy(d_columnFlags, d_fiberStartFlags + level * nnz, nnz * sizeof(int64_t), cudaMemcpyDeviceToDevice);

//         // Compute prefix sum
//         int64_t* d_columnResults;
//         cudaMalloc(&d_columnResults, nnz * sizeof(int64_t));
//         computePrefixSumNonZero(d_columnFlags, d_columnResults, nnz);

//         // Copy results back to the output
//         cudaMemcpy(d_prefixSumResults + level * nnz, d_columnResults, nnz * sizeof(int64_t), cudaMemcpyDeviceToDevice);

//         cudaFree(d_columnFlags);
//         cudaFree(d_columnResults);
//     }
// }




// Compute prefix sum for specific orders
void computePrefixSumPerOrder(thrust::device_vector<int64_t>& fiberStartFlags, int order, int64_t nnz,
                               thrust::device_vector<int64_t>& prefixSumResults) {
    // Create a temporary vector for results
    thrust::device_vector<int64_t> tempFlags(fiberStartFlags);

    // Extract raw pointers from the device vectors
    int64_t* fiberStartFlags_ptr = thrust::raw_pointer_cast(fiberStartFlags.data());
    int64_t* tempFlags_ptr = thrust::raw_pointer_cast(tempFlags.data());

    // Iterate for each order (column) and compute the prefix sum
    for (int level = 0; level < order; ++level) {
        // Create a temporary array for the current column
        thrust::device_vector<int64_t> columnFlags(nnz, 0);
        int64_t* columnFlags_ptr = thrust::raw_pointer_cast(columnFlags.data());

        // Copy the column values to the temporary array
        thrust::for_each(
            thrust::make_counting_iterator<int64_t>(0),
            thrust::make_counting_iterator<int64_t>(nnz),
            [=] __device__(int64_t idx) {
                if (idx < nnz) {
                    columnFlags_ptr[idx] = fiberStartFlags_ptr[idx * order + level] ;
                }
            });


        // cout << "columnFlags_ptr : " << "level : " << level << " " << columnFlags_ptr[1] << endl;
        // cout << "fiberStartFlags_ptr : " << "level : " << level << " " << fiberStartFlags_ptr[1] << endl;
        

        // Compute prefix sum on the column
        thrust::inclusive_scan(columnFlags.begin(), columnFlags.end(), columnFlags.begin());

        // Copy the results back to tempFlags
        thrust::for_each(
            thrust::make_counting_iterator<int64_t>(0),
            thrust::make_counting_iterator<int64_t>(nnz),
            [=] __device__(int64_t idx) {
                if (idx < nnz) {
                    tempFlags_ptr[idx * order + level] = columnFlags_ptr[idx];
                }
            });
    }

    // Copy the results to prefixSumResults
    prefixSumResults = tempFlags;
}






// Kernel to populate idx and ptr arrays
__global__ void fill_idx_ptr(const COOElement* sortedCOO, int64_t nnz, int order,
                             const int64_t* prefixSumResults, int64_t* idx, int64_t* ptr,
                             double* vals, int64_t level, int64_t* tempCounts, int64_t* count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;

    if (level == order - 1) {
        vals[tid] = sortedCOO[tid].value;
    }

    if (tid == 0) {
        ptr[0] = 0;  // First element of ptr array
    }

    if (level == 0) {
        //printf("Thread id : %d, count : %lld\n", tid, *count);
        if(tid == 0){
            atomicAddInt64(&tempCounts[0], 1);
            //int idxPosition = fiberStartFlags[order * tid + level] - 1;
            idx[0] = sortedCOO[0].indices[level];
        }
        else{
            if (prefixSumResults[order * tid + level] != prefixSumResults[order * (tid - 1) + level]) {
                //int idxPosition = atomicAddInt64(&tempCounts[0], 1);
                atomicAddInt64(&tempCounts[0], 1);
                int idxPosition = prefixSumResults[order * tid + level] - 1;
                idx[idxPosition] = sortedCOO[tid].indices[level];
                // atomicAddInt64(count, 1);
            }
        }
        //printf("Thread id : %d, count : %lld\n", tid, *count);

        // if (tid == nnz - 1) {
        //     ptr[1] = *count;
        // }
    } 
    else {
        if(tid == 0){
            atomicAddInt64(&tempCounts[0], 1);
            //int idxPosition = fiberStartFlags[order * tid + level] - 1;
            idx[0] = sortedCOO[0].indices[level];

            int tid_copy = tid + 1;

            int ptrPosition = 0, ptr_arr_val = 1;

            while (tid_copy < nnz && prefixSumResults[order * tid_copy + level - 1] == prefixSumResults[order * (tid_copy - 1) + level - 1]) {
                if (prefixSumResults[order * tid_copy + level] != prefixSumResults[order * (tid_copy - 1) + level]) {
                    atomicAddInt64(&tempCounts[0], 1);
                    ptr_arr_val = prefixSumResults[order * tid_copy + level];
                    int idxPosition = prefixSumResults[order * tid_copy + level] - 1;
                    idx[idxPosition] = sortedCOO[tid_copy].indices[level];
                    // atomicAddInt64(count, 1);
                }
                tid_copy++;
            }

            ptrPosition= prefixSumResults[order * (tid_copy - 1) + level - 1];
            atomicAddInt64(&tempCounts[1], 1);
            ptr[ptrPosition] = ptr_arr_val;

        }
        else if(prefixSumResults[order * tid + level - 1] != prefixSumResults[order * (tid - 1) + level - 1]) {
            atomicAddInt64(&tempCounts[0], 1);
            int idxPosition = prefixSumResults[order * tid + level] - 1;
            idx[idxPosition] = sortedCOO[tid].indices[level];
            //atomicAddInt64(count, 1);

            int ptrPosition = 0, ptr_arr_val = prefixSumResults[order * tid + level];

            int tid_copy = tid + 1;

            while (tid_copy < nnz && prefixSumResults[order * tid_copy + level - 1] == prefixSumResults[order * (tid_copy - 1) + level - 1]) {
                if (prefixSumResults[order * tid_copy + level] != prefixSumResults[order * (tid_copy - 1) + level]) {
                    atomicAddInt64(&tempCounts[0], 1);
                    ptr_arr_val = prefixSumResults[order * tid_copy + level];
                    int idxPosition = prefixSumResults[order * tid_copy + level] - 1;
                    idx[idxPosition] = sortedCOO[tid_copy].indices[level];
                    //atomicAddInt64(count, 1);
                }
                tid_copy++;
            }

            // Update ptr for the next level
            if(tid_copy < nnz){
                ptrPosition = prefixSumResults[order * (tid_copy - 1) + level - 1];
            }
            else{
                ptrPosition = prefixSumResults[order * (nnz - 1) + level - 1];
            }
            // ptrPosition = prefixSumResults[order * (tid_copy - 1) + level - 1];
            atomicAddInt64(&tempCounts[1], 1);
            ptr[ptrPosition] = ptr_arr_val;
            if(ptrPosition == 10){
                printf("ptrPosition :  %d ptr_arr_val : %d\n", ptrPosition, ptr_arr_val);
            }
            //printf("Hello from thread %d in block %d\n", threadIdx.x, blockIdx.x);
            // if(level == 3 && tid == nnz - 1){
            //     ptr[ptrPosition] = 55;
            // }
            // else{
            //     ptr[ptrPosition] = ptr_arr_val;
            // }
        }
    }
}


// Lexicographic comparator for COOElement
struct COOComparator {
    int order;

    COOComparator(int order) : order(order) {}

    __host__ __device__ bool operator()(const COOElement& a, const COOElement& b) const {
        for (int i = 0; i < order; ++i) {
            if (a.indices[i] < b.indices[i]) return true;
            if (a.indices[i] > b.indices[i]) return false;
        }
        return false;
    }
};

void convertCOOtoCSF_GPU(const std::vector<COOElement>& cooData, int order,
                         std::vector<std::vector<int64_t>>& idx_gpu, std::vector<std::vector<int64_t>>& ptr_gpu, 
                         std::vector<double>& vals_gpu) {
    int64_t nnz = cooData.size();

    // Host to device transfer
    thrust::device_vector<COOElement> d_cooData(cooData);
    thrust::device_vector<int64_t> d_fiberStartFlags(order * nnz, 0);
    thrust::device_vector<double> d_vals(nnz);
    thrust::device_vector<int64_t> d_counts(order, 0);

    // Sort the COO data lexicographically
    thrust::sort(thrust::device, d_cooData.begin(), d_cooData.end(), COOComparator(order));

    // Now print sorted COO data
    thrust::host_vector<COOElement> h_sortedCOO = d_cooData;

    std::cout << "Sorted COO Data:" << std::endl;

    for (const auto& elem : h_sortedCOO) {
        for (int i = 0; i < order; ++i) std::cout << elem.indices[i] << " ";
        std::cout << elem.value << std::endl;
    }

    // Allocate idx and ptr arrays for each level
    std::vector<thrust::device_vector<int64_t>> d_idx(order);
    std::vector<thrust::device_vector<int64_t>> d_ptr(order);

    // for (int i = 0; i < order; ++i) {
    //     d_idx[i].resize(nnz);  // Maximum size for idx
    //     d_ptr[i].resize(nnz + 1);  // Maximum size for ptr
    // }

    // Launch kernel to detect fibers
    // int blockSize_detect_fibers = 256;
    // int numBlocks_detect_fibers = (nnz + blockSize_detect_fibers - 1) / blockSize_detect_fibers;


    int blockSize = 256;
    int numBlocks = (nnz + blockSize - 1) / blockSize;

    detect_fibers<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_cooData.data()), nnz, order,
                                            thrust::raw_pointer_cast(d_fiberStartFlags.data()), thrust::raw_pointer_cast(d_counts.data()));


    
    
    thrust::host_vector<int64_t> h_fiberStartFlags = d_fiberStartFlags;
    std::cout << "Fiber Start Flags:" << std::endl;

    for (size_t i = 0; i < nnz; ++i) {
        for (int j = 0; j < order; ++j) {
            std::cout << h_fiberStartFlags[i * order + j] << " ";
        }
        std::cout << std::endl;
    }

    thrust::host_vector<int64_t> h_counts = d_counts;
    std::cout << "Ptr and Idx sizes :" << std::endl;
    for(int level = 0; level < order; level++){
        d_idx[level].resize(h_counts[level]);
        d_ptr[level].resize(h_counts[level] + 1);
        cout << d_ptr[level].size() << " " << d_idx[level].size() << endl;
    }


    thrust::device_vector<int64_t> d_prefixSumResults(order * nnz);
    computePrefixSumPerOrder(d_fiberStartFlags, order, nnz, d_prefixSumResults);


    // Method - 2 for calculating 
    // computePrefixSumPerOrder(
    //     thrust::raw_pointer_cast(d_fiberStartFlags.data()),
    //     order,
    //     nnz,
    //     thrust::raw_pointer_cast(d_prefixSumResults.data())
    // );



    // Optional: Copy results back to the host for debugging
    thrust::host_vector<int64_t> h_prefixSumResults = d_prefixSumResults;
    std::cout << "Prefix Sum Results:" << std::endl;
    for (int64_t i = 0; i < nnz; ++i) {
        for (int level = 0; level < order; ++level) {
            std::cout << h_prefixSumResults[i * order + level] << " ";
        }
        std::cout << std::endl;
    }


    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after detect_fibers: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }

    idx_gpu.resize(order);
    ptr_gpu.resize(order);

    // int blockSize_fill_idx_ptr = order;
    // int numBlocks_fill_idx_ptr = (nnz + blockSize_fill_idx_ptr - 1) / blockSize_fill_idx_ptr;


    for (int level = 0; level < order; ++level) {
        // Temporary counters for idx and ptr positions
        thrust::device_vector<int64_t> d_tempCounts(2, 0);
        d_tempCounts[1] = 1;

        int64_t h_count = 0;  // Host-side count variable
        int64_t* d_count;     // Device-side count pointer

        cudaMalloc(&d_count, sizeof(int64_t));
        cudaMemcpy(d_count, &h_count, sizeof(int64_t), cudaMemcpyHostToDevice);

        // Launch kernel to populate idx and ptr for the current level
        fill_idx_ptr<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_cooData.data()), nnz, order,
                                               thrust::raw_pointer_cast(d_prefixSumResults.data()),
                                               thrust::raw_pointer_cast(d_idx[level].data()),
                                               thrust::raw_pointer_cast(d_ptr[level].data()),
                                               thrust::raw_pointer_cast(d_vals.data()), level,
                                               thrust::raw_pointer_cast(d_tempCounts.data()), d_count);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error after fill_idx_ptr: " << cudaGetErrorString(err) << std::endl;
            exit(-1);
        }

        // Retrieve actual sizes from d_tempCounts
        int64_t h_tempCounts[2];
        cudaMemcpy(h_tempCounts, thrust::raw_pointer_cast(d_tempCounts.data()), 
                2 * sizeof(int64_t), cudaMemcpyDeviceToHost);

        if(level == 0){
            //*(d_ptr[level].begin() + 1) = h_tempCounts[0];

            //cout << h_tempCounts[0] << endl;
            //cout << h_tempCounts[1] << endl;
            // ptr_gpu[0].resize(2);
            d_ptr[0][1] = h_tempCounts[0];
            
            //cout << d_ptr[0][1] << endl;
        }

        // Resize idx and ptr vectors on the host and Copy the resized data from device to host
        idx_gpu[level].resize(h_tempCounts[0]);
        thrust::copy(d_idx[level].begin(), d_idx[level].begin() + h_tempCounts[0], idx_gpu[level].begin());
        
        if(level == 0){
            cout << "idx size  : " <<  h_tempCounts[0] << endl;
            cout << "ptr size  : " << h_tempCounts[1] << endl;
            ptr_gpu[0].resize(2);
            thrust::copy(d_ptr[level].begin(), d_ptr[level].begin() + 2, ptr_gpu[level].begin());
        }
        else{
            cout << "idx size  : " <<  h_tempCounts[0] << endl;
            cout << "ptr size  : " << h_tempCounts[1] << endl;
            ptr_gpu[level].resize(h_tempCounts[1]);
            d_ptr[level].resize(h_tempCounts[1]);
            thrust::copy(d_ptr[level].begin(), d_ptr[level].begin() + h_tempCounts[1], ptr_gpu[level].begin());
        }

        //cout << ptr_gpu[0][0] << " " << ptr_gpu[0][1] << endl;

        // Copy the resized data from device to host
        
        //thrust::copy(d_ptr[level].begin(), d_ptr[level].begin() + h_tempCounts[1], ptr_gpu[level].begin());

        //cout << ptr_gpu[0][0] << " " << ptr_gpu[0][1] << endl;

        cudaFree(d_count);
    }

    // Copy results back to host
    vals_gpu.resize(nnz);
    thrust::copy(d_vals.begin(), d_vals.end(), vals_gpu.begin());

    // idx_gpu.resize(order);
    // ptr_gpu.resize(order);
    // for (int i = 0; i < order; ++i) {
    //     idx_gpu[i].resize(d_idx[i].size());
    //     ptr_gpu[i].resize(d_ptr[i].size());
    //     thrust::copy(d_idx[i].begin(), d_idx[i].end(), idx_gpu[i].begin());
    //     thrust::copy(d_ptr[i].begin(), d_ptr[i].end(), ptr_gpu[i].begin());
    // }
}


// Function to read COO data from file
vector<COOElement> readCOO(const string& filename, int& order) {
    vector<COOElement> cooData;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        istringstream ss(line);
        vector<int64_t> indices;
        double value;
        int64_t index;

        // Read indices
        while (ss >> index) {
            indices.push_back(index);
        }

        // Last element is the value
        value = indices.back();
        indices.pop_back(); // Remove value from indices vector

        // Detect the order based on the number of indices
        if (cooData.empty()) {
            order = indices.size(); // Tensor order
        }

        COOElement element;
        std::copy(indices.begin(), indices.begin() + order, element.indices); // If `indices` is an array
        element.value = value;
        cooData.push_back(element);
    }

    file.close();
    return cooData;
}


/*
Command to Compile the file : 
nvcc -o cootocsfgpu COO_to_CSF_GPU.cu


*/

int main(int argc, char* argv[]) {
    // // Check for the correct number of arguments
    // if (argc < 4) {
    //     cerr << "Usage: " << argv[0] << " <order> <dim_0> <dim_1> <dim_2> ..." << endl;
    //     return 1;
    // }



    // // Save the first four arguments
    // int order = atoi(argv[1]);
    // int dim_0 = atoi(argv[2]);
    // int dim_1 = atoi(argv[3]);
    // int dim_2 = atoi(argv[4]);

    // int64_t* my_tensor_indices = nullptr;
    // double* my_tensor_values = nullptr;
    // int total_indices = 0;
    // int total_values = 0;

    // generate_tensor(argc, argv, &my_tensor_indices, &my_tensor_values, &total_indices, &total_values);

    // vector<COOElement> cooData(total_values); // Initialize with number of non-zero values
    // for(int i=0, j=0; i < total_indices; i++) {
    //     cooData[j].indices.push_back(my_tensor_indices[i]-1);
    //     if((i + 1) % order == 0) {
    //         cooData[j].value = my_tensor_values[j]; // Assign the value after indices
    //         j++;
    //     }
    // }



    // cout << "Order of the Tensor : " << order << endl;
    // cout << "Dimension - 0 : " << dim_0 << endl;
    // cout << "Dimension - 1 : " << dim_1 << endl;
    // cout << "Dimension - 2 : " << dim_2 << endl;
    // cout << "Total size of my_tensor_indices : " << total_indices<< endl;
    // cout << "Total size of my_tensor_values : " << total_values << endl;

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


    string filename = "coo.txt"; // Input file containing COO data
    int order = 3; // Tensor order
    vector<COOElement> cooData = readCOO(filename, order);

    cout << "Order of the Tensor: " << order << endl;
    cout << "Tensor in COO Format:\n";
    for (const auto& elem : cooData) {
        for (auto idx : elem.indices) cout << idx << " ";
        cout << elem.value << endl;
    }

    std::vector<std::vector<int64_t>> idx_gpu, ptr_gpu;
    std::vector<double> vals_gpu;

    convertCOOtoCSF_GPU(cooData, order, idx_gpu, ptr_gpu, vals_gpu);

    // Print the results
    for (int i = 0; i < order; ++i) {
        std::cout << "mode_" << i << "_ptr: ";
        for (auto v : ptr_gpu[i]) std::cout << v << " ";
        std::cout << "\nmode_" << i << "_idx: ";
        for (auto v : idx_gpu[i]) std::cout << v << " ";
        std::cout << "\n";
    }

    std::cout << "vals: ";
    for (auto v : vals_gpu) std::cout << v << " ";
    std::cout << std::endl;

    return 0;
}
