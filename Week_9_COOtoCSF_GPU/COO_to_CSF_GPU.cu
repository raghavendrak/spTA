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

struct COOElement {
    int64_t indices[3];  // Store indices for all modes
    double value;
};

int64_t *fiberStartFlags;
COOElement *sortedCOO;

// Kernel to detect fibers and populate fiber start flags
__global__ void detect_fibers(const COOElement* cooData, int64_t nnz, int order, int64_t* fiberStartFlags) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;

    if (tid == 0) {
        for (int level = 0; level < order; ++level) {
            fiberStartFlags[order * tid + level] = 1;  // First element is always a fiber start
        }
    } else {
        for (int level = 0; level < order; ++level) {
            if (cooData[tid].indices[level] != cooData[tid - 1].indices[level]) {
                fiberStartFlags[order * tid + level] = 1;
            }
        }
    }
}

__device__ inline int64_t atomicAddInt64(int64_t* address, int64_t val) {
    return atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

// Kernel to populate idx and ptr arrays
__global__ void fill_idx_ptr(const COOElement* sortedCOO, int64_t nnz, int order,
                             const int64_t* fiberStartFlags, int64_t* idx, int64_t* ptr,
                             double* vals, int64_t level, int64_t* tempCounts, int64_t* count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;

    if (level == order) {
        vals[tid] = sortedCOO[tid].value;
    }

    if (tid == 0) {
        ptr[0] = 0;  // First element of ptr array
    }

    if (level == 0) {
        if (fiberStartFlags[order * tid + level]) {
            int idxPosition = atomicAddInt64(&tempCounts[0], 1);
            idx[idxPosition] = sortedCOO[tid].indices[level];
            atomicAddInt64(count, 1);
        }

        if (tid == nnz - 1) {
            ptr[1] = *count;
        }
    } else {
        if (fiberStartFlags[order * tid + level - 1]) {
            int idxPosition = atomicAddInt64(&tempCounts[0], 1);
            idx[idxPosition] = sortedCOO[tid].indices[level];
            atomicAddInt64(count, 1);

            int tid_copy = tid + 1;

            while (tid_copy < nnz && fiberStartFlags[order * tid_copy + level - 1] == 0) {
                if (fiberStartFlags[order * tid_copy + level]) {
                    int idxPosition = atomicAddInt64(&tempCounts[0], 1);
                    idx[idxPosition] = sortedCOO[tid_copy].indices[level];
                    atomicAddInt64(count, 1);
                }
                tid_copy++;
            }

            // Update ptr for the next level
            int ptrPosition = atomicAddInt64(&tempCounts[1], 1);
            ptr[ptrPosition] = ptr[ptrPosition - 1] + *count;
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

    // Sort the COO data lexicographically
    thrust::sort(thrust::device, d_cooData.begin(), d_cooData.end(), COOComparator(order));

    // Allocate idx and ptr arrays for each level
    std::vector<thrust::device_vector<int64_t>> d_idx(order);
    std::vector<thrust::device_vector<int64_t>> d_ptr(order);

    for (int i = 0; i < order; ++i) {
        d_idx[i].resize(nnz);  // Maximum size for idx
        d_ptr[i].resize(nnz + 1);  // Maximum size for ptr
    }

    // Launch kernel to detect fibers
    int blockSize = 256;
    int numBlocks = (nnz + blockSize - 1) / blockSize;

    detect_fibers<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_cooData.data()), nnz, order,
                                            thrust::raw_pointer_cast(d_fiberStartFlags.data()));

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after detect_fibers: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }

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
                                               thrust::raw_pointer_cast(d_fiberStartFlags.data()),
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

        cudaMemcpy(&h_count, d_count, sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaFree(d_count);
    }

    // Copy results back to host
    vals_gpu.resize(nnz);
    thrust::copy(d_vals.begin(), d_vals.end(), vals_gpu.begin());

    idx_gpu.resize(order);
    ptr_gpu.resize(order);
    for (int i = 0; i < order; ++i) {
        idx_gpu[i].resize(d_idx[i].size());
        ptr_gpu[i].resize(d_ptr[i].size());
        thrust::copy(d_idx[i].begin(), d_idx[i].end(), idx_gpu[i].begin());
        thrust::copy(d_ptr[i].begin(), d_ptr[i].end(), ptr_gpu[i].begin());
    }
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
        std::copy(indices.begin(), indices.begin() + 3, element.indices); // If `indices` is an array
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
