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
#include "genten_gpu.h"
#include <chrono>
#include <string>
#include <cstdint>
#include <cmath> // For comparing floating point numbers with tolerance

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
        if(tid == 0){
            atomicAddInt64(&tempCounts[0], 1);
            idx[0] = sortedCOO[0].indices[level];
        }
        else{
            if (prefixSumResults[order * tid + level] != prefixSumResults[order * (tid - 1) + level]) {
                atomicAddInt64(&tempCounts[0], 1);
                int idxPosition = prefixSumResults[order * tid + level] - 1;
                idx[idxPosition] = sortedCOO[tid].indices[level];
            }
        }
    } 
    else {
        if(tid == 0){
            atomicAddInt64(&tempCounts[0], 1);
            idx[0] = sortedCOO[0].indices[level];

            int tid_copy = tid + 1;

            int ptrPosition = 0, ptr_arr_val = 1;

            while (tid_copy < nnz && prefixSumResults[order * tid_copy + level - 1] == prefixSumResults[order * (tid_copy - 1) + level - 1]) {
                if (prefixSumResults[order * tid_copy + level] != prefixSumResults[order * (tid_copy - 1) + level]) {
                    atomicAddInt64(&tempCounts[0], 1);
                    ptr_arr_val = prefixSumResults[order * tid_copy + level];
                    int idxPosition = prefixSumResults[order * tid_copy + level] - 1;
                    idx[idxPosition] = sortedCOO[tid_copy].indices[level];
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

            int ptrPosition = 0, ptr_arr_val = prefixSumResults[order * tid + level];

            int tid_copy = tid + 1;

            while (tid_copy < nnz && prefixSumResults[order * tid_copy + level - 1] == prefixSumResults[order * (tid_copy - 1) + level - 1]) {
                if (prefixSumResults[order * tid_copy + level] != prefixSumResults[order * (tid_copy - 1) + level]) {
                    atomicAddInt64(&tempCounts[0], 1);
                    ptr_arr_val = prefixSumResults[order * tid_copy + level];
                    int idxPosition = prefixSumResults[order * tid_copy + level] - 1;
                    idx[idxPosition] = sortedCOO[tid_copy].indices[level];
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
            atomicAddInt64(&tempCounts[1], 1);
            ptr[ptrPosition] = ptr_arr_val;
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

    // std::cout << "Sorted COO Data:" << std::endl;

    // for (const auto& elem : h_sortedCOO) {
    //     for (int i = 0; i < order; ++i) std::cout << elem.indices[i] << " ";
    //     std::cout << elem.value << std::endl;
    // }

    // Allocate idx and ptr arrays for each level
    std::vector<thrust::device_vector<int64_t>> d_idx(order);
    std::vector<thrust::device_vector<int64_t>> d_ptr(order);

    int blockSize = 256;
    int numBlocks = (nnz + blockSize - 1) / blockSize;

    auto start_gpu = std::chrono::high_resolution_clock::now();

    detect_fibers<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_cooData.data()), nnz, order,
                                            thrust::raw_pointer_cast(d_fiberStartFlags.data()), thrust::raw_pointer_cast(d_counts.data()));


    
    
    thrust::host_vector<int64_t> h_fiberStartFlags = d_fiberStartFlags;
    //std::cout << "Fiber Start Flags:" << std::endl;

    // for (size_t i = 0; i < nnz; ++i) {
    //     for (int j = 0; j < order; ++j) {
    //         std::cout << h_fiberStartFlags[i * order + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    thrust::host_vector<int64_t> h_counts = d_counts;
    //std::cout << "Ptr and Idx sizes :" << std::endl;
    for(int level = 0; level < order; level++){
        d_idx[level].resize(h_counts[level]);
        d_ptr[level].resize(h_counts[level] + 1);
        //cout << d_ptr[level].size() << " " << d_idx[level].size() << endl;
    }


    thrust::device_vector<int64_t> d_prefixSumResults(order * nnz);
    computePrefixSumPerOrder(d_fiberStartFlags, order, nnz, d_prefixSumResults);

    // Optional: Copy results back to the host for debugging
    thrust::host_vector<int64_t> h_prefixSumResults = d_prefixSumResults;
    // std::cout << "Prefix Sum Results:" << std::endl;
    // for (int64_t i = 0; i < nnz; ++i) {
    //     for (int level = 0; level < order; ++level) {
    //         std::cout << h_prefixSumResults[i * order + level] << " ";
    //     }
    //     std::cout << std::endl;
    // }


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
            d_ptr[0][1] = h_tempCounts[0];
        }

        // Resize idx and ptr vectors on the host and Copy the resized data from device to host
        idx_gpu[level].resize(h_tempCounts[0]);
        thrust::copy(d_idx[level].begin(), d_idx[level].begin() + h_tempCounts[0], idx_gpu[level].begin());
        
        if(level == 0){
            ptr_gpu[0].resize(2);
            thrust::copy(d_ptr[level].begin(), d_ptr[level].begin() + 2, ptr_gpu[level].begin());
        }
        else{
            ptr_gpu[level].resize(h_tempCounts[1]);
            d_ptr[level].resize(h_tempCounts[1]);
            thrust::copy(d_ptr[level].begin(), d_ptr[level].begin() + h_tempCounts[1], ptr_gpu[level].begin());
        }

        cudaFree(d_count);
    }

    // Copy results back to host
    vals_gpu.resize(nnz);
    thrust::copy(d_vals.begin(), d_vals.end(), vals_gpu.begin());

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    std::cout << "COO to CSF GPU execution time: " << gpu_time.count() << " seconds." << std::endl;

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


// Functions for COOtoCSF_CPU : 

// Structure to hold the COO data
struct COOElement_CPU {
    vector<int64_t> indices; // Store indices for all modes
    double value;
};


// Function to convert COO to CSF format dynamically
void convertCOOtoCSF_CPU(const vector<COOElement_CPU>& cooData_CPU, int order, vector<std::vector<int64_t>> &idx_cpu, vector<std::vector<int64_t>> &ptr_cpu, vector<double>& vals_cpu) {
    int64_t nnz = cooData_CPU.size();
    
    // CSF structure
    vector<vector<int64_t>> idx(order); // `order` levels of indices
    vector<vector<int64_t>> ptr(order); // `order` levels of pointers
    vector<double> vals;   // Store non-zero values

    
    
    // Initialize mode_0_ptr array with '0'
    ptr[0].push_back(0); 
    
    
    // Sorting the COO data lexicographically by all indices
    vector<COOElement_CPU> sortedCOO_CPU = cooData_CPU;
    sort(sortedCOO_CPU.begin(), sortedCOO_CPU.end(), [&](const COOElement_CPU& a, const COOElement_CPU& b) {
        for (int i = 0; i < order; ++i) {
            if (a.indices[i] != b.indices[i]) {
                return a.indices[i] < b.indices[i];
            }
        }
        return false;
    });
    
    // Variables to track the previous indices at each level
    vector<int64_t> prevIndices(order, -1);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    // Loop over the sorted COO data
    for (const auto& elem : sortedCOO_CPU) {
        int flag = 1;   // Flag is for indicating the fiber split
        for (int level = 0; level < order; ++level) {            
            if (elem.indices[level] != prevIndices[level] || flag == 0) {
                flag = 0;

                // New index for this level
                idx[level].push_back(elem.indices[level]);

                // Once there is a fiber split, push the current size of idx[level + 1] in the ptr[level + 1] vector
                if (level + 1 < order) {
                    ptr[level+1].push_back(idx[level+1].size());
                }
                
                // Update the previous index for this level
                prevIndices[level] = elem.indices[level];
            }
        }

        // Store the value for this element
        vals.push_back(elem.value);
    }

    // Finalize pointers: the last value in all pointer vectors is the size of the corresponding index array
    for (int level = 0; level < order; ++level){
        ptr[level].push_back(idx[level].size());  // Final entry to mark the end
    }

    idx_cpu = idx;
    ptr_cpu = ptr;
    vals_cpu = vals;

    

    // // Transfer CSF data dynamically based on the detected order
    // for (int level = 0; level + 1 < order; ++level) {
    //     //cout << "mode_" << level << "_ptr : ";
    //     for (auto val : ptr[level]) {
    //         ptr_cpu[level].push_back(val);
    //     }
    //     for (auto val : idx[level]) {
    //         idx_cpu[level].push_back(val);
    //     }
    // }

    // // Transfer the values
    // for (auto val : vals) {
    //     vals_cpu.push_back(val);
    // }

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    std::cout << "COO to CSF CPU execution time: " << cpu_time.count() << " seconds." << std::endl;
}


// Functions for comparing both CPU and GPU results : 

// Function to compare two vectors
bool compareVectors(const std::vector<int64_t>& vec1, const std::vector<int64_t>& vec2) {
    if (vec1.size() != vec2.size()) {
        return false;
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        if (vec1[i] != vec2[i]) {
            return false;
        }
    }

    return true;
}

// Function to compare two vectors of doubles with a tolerance
bool compareValues(const std::vector<double>& vec1, const std::vector<double>& vec2, double tolerance = 1e-6) {
    if (vec1.size() != vec2.size()) {
        return false;
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::fabs(vec1[i] - vec2[i]) > tolerance) {
            return false;
        }
    }

    return true;
}

// Function to compare CPU and GPU results
void compareResults(const std::vector<std::vector<int64_t>>& idx_gpu,
                    const std::vector<std::vector<int64_t>>& idx_cpu,
                    const std::vector<std::vector<int64_t>>& ptr_gpu,
                    const std::vector<std::vector<int64_t>>& ptr_cpu,
                    const std::vector<double>& vals_gpu,
                    const std::vector<double>& vals_cpu,
                    int order) {
    
    bool idxMatch = true, ptrMatch = true, valsMatch = true;

    // Compare idx for all levels
    for (int level = 0; level < order; ++level) {
        if (!compareVectors(idx_gpu[level], idx_cpu[level])) {
            idxMatch = false;
            std::cout << "Mismatch in idx at level " << level << "\n";
        }
    }

    // Compare ptr for all levels
    for (int level = 0; level < order; ++level) {
        if (!compareVectors(ptr_gpu[level], ptr_cpu[level])) {
            ptrMatch = false;
            std::cout << "Mismatch in ptr at level " << level << "\n";
        }
    }

    // Compare vals
    if (!compareValues(vals_gpu, vals_cpu)) {
        valsMatch = false;
        std::cout << "Mismatch in vals\n";
    }

    // Output the final result
    if (idxMatch && ptrMatch && valsMatch) {
        std::cout << "CPU and GPU results are matching\n";
    } else {
        std::cout << "CPU and GPU results are not matching\n";
    }
}



/*
Command to Compile the file : 
1st command : 
gcc -O2 -Wall -fopenmp -c genten.c -o genten.o -lm

2nd command :
nvcc -o cootocsfgpu COO_to_CSF_GPU.cu genten.o -Xcompiler -fopenmp --extended-lambda

Command to run the file : 
./cootocsfgpu 3 2000 2000 2000 -d 0.01 -f 0.1 -c 0.5 -v 0.5
*/

int main(int argc, char* argv[]) {
    // Check for the correct number of arguments
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <order> <dim_0> <dim_1> <dim_2> ..." << endl;
        return 1;
    }



    // Save the first four arguments
    int order = atoi(argv[1]);
    int dim_0 = atoi(argv[2]);
    int dim_1 = atoi(argv[3]);
    int dim_2 = atoi(argv[4]);

    int64_t* my_tensor_indices = nullptr;
    double* my_tensor_values = nullptr;
    int total_indices = 0;
    int total_values = 0;

    generate_tensor(argc, argv, &my_tensor_indices, &my_tensor_values, &total_indices, &total_values);

    int index_counter = 0;
    vector<COOElement> cooData(total_values); // Initialize with number of non-zero values
    for(int i=0, j=0; i < total_indices; i++) {
        cooData[j].indices[index_counter] = my_tensor_indices[i]-1;
        index_counter++;

        if (index_counter == order){
            cooData[j].value = my_tensor_values[j]; // Assign the value after indices
            index_counter = 0; // Reset the counter for the next COOElement
            j++;
        }
    }



    cout << "Order of the Tensor : " << order << endl;
    cout << "Dimension - 0 : " << dim_0 << endl;
    cout << "Dimension - 1 : " << dim_1 << endl;
    cout << "Dimension - 2 : " << dim_2 << endl;
    cout << "Total size of my_tensor_indices : " << total_indices<< endl;
    cout << "Total size of my_tensor_values : " << total_values << endl;

    // cout << "Tensor in COO Format : " << endl;
    // for(int i=0, j=0; i<total_indices; i++){
    //     cout << my_tensor_indices[i] << " ";
    //     if((i+1)%order == 0){
    //         cout << my_tensor_values[j];
    //         j++;
    //         cout << endl;
    //     }
    // }
    // cout << endl;


    //string filename = "coo.txt"; // Input file containing COO data
    //int order = 3; // Tensor order
    //vector<COOElement> cooData = readCOO(filename, order);

    // cout << "Order of the Tensor: " << order << endl;
    // cout << "Tensor in COO Format:\n";
    // for (const auto& elem : cooData) {
    //     for (auto idx : elem.indices) cout << idx << " ";
    //     cout << elem.value << endl;
    // }

    std::vector<std::vector<int64_t>> idx_gpu, ptr_gpu;
    std::vector<double> vals_gpu;

    convertCOOtoCSF_GPU(cooData, order, idx_gpu, ptr_gpu, vals_gpu);

    // Print the results
    // for (int level = 0; level < order; ++level) {
    //     std::cout << "mode_" << level << "_ptr: ";
    //     for (auto v : ptr_gpu[level]) std::cout << v << " ";
    //     std::cout << "\nmode_" << level << "_idx: ";
    //     for (auto v : idx_gpu[level]) std::cout << v << " ";
    //     std::cout << "\n";
    // }

    // std::cout << "vals: ";
    // for (auto v : vals_gpu) std::cout << v << " ";
    // std::cout << std::endl;

    std::vector<std::vector<int64_t>> idx_cpu, ptr_cpu;
    std::vector<double> vals_cpu;

    vector<COOElement_CPU> cooData_CPU(total_values); // Initialize with number of non-zero values
    for(int i=0, j=0; i < total_indices; i++) {
        cooData_CPU[j].indices.push_back(my_tensor_indices[i]-1);

        if ((i + 1) % order == 0){
            cooData_CPU[j].value = my_tensor_values[j]; // Assign the value after indices
            j++;
        }
    }

    convertCOOtoCSF_CPU(cooData_CPU, order, idx_cpu, ptr_cpu, vals_cpu);

    // Print the results
    // for (int level = 0; level < order; ++level) {
    //     std::cout << "mode_" << level << "_ptr: ";
    //     for (auto v : ptr_cpu[level]) std::cout << v << " ";
    //     std::cout << "\nmode_" << level << "_idx: ";
    //     for (auto v : idx_cpu[level]) std::cout << v << " ";
    //     std::cout << "\n";
    // }

    // std::cout << "vals: ";
    // for (auto v : vals_cpu) std::cout << v << " ";
    // std::cout << std::endl;

    // After generating the GPU and CPU results (idx_gpu, ptr_gpu, vals_gpu, idx_cpu, ptr_cpu, vals_cpu)
    compareResults(idx_gpu, idx_cpu, ptr_gpu, ptr_cpu, vals_gpu, vals_cpu, order);

    return 0;
}
