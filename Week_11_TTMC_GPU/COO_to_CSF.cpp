// Example : 

// Input COO Data : 

// last element represent the value of the non-zero element and the remaining elements in each line represents the indices

// 1 1 1 5
// 1 2 2 5
// 2 1 1 5
// 2 1 2 5
// 2 1 3 5
// 2 1 4 5



// Output : 
// mode_0_ptr : 0 2 
// mode_0_idx : 1 2 

// mode_1_ptr : 0 2 3 
// mode_1_idx : 1 2 1 

// mode_2_ptr : 0 1 2 6 
// mode_2_idx : 1 2 1 2 3 4 

// vals : 5 5 5 5 5 5


#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <sstream>
#include <cstdint>  

using namespace std;

//int max_size = 100000; 

uint64_t* mode_0_pointer_csf;
uint64_t* mode_0_indices_csf;
uint64_t* mode_1_pointer_csf;
uint64_t* mode_1_indices_csf;
uint64_t* mode_2_pointer_csf;
uint64_t* mode_2_indices_csf;

int size_mode_0_pointer_csf = 0, size_mode_0_indices_csf = 0;
int size_mode_1_pointer_csf = 0, size_mode_1_indices_csf = 0;
int size_mode_2_pointer_csf = 0, size_mode_2_indices_csf = 0;

void get_mode_0_ptr(uint64_t** mode_0_ptr_ttmc, int* size_mode_0_ptr){
    (*mode_0_ptr_ttmc) = mode_0_pointer_csf ;
    *size_mode_0_ptr = size_mode_0_pointer_csf;
}
void get_mode_0_idx(uint64_t** mode_0_idx_ttmc, int* size_mode_0_idx){
    (*mode_0_idx_ttmc) = mode_0_indices_csf ;
    *size_mode_0_idx = size_mode_0_indices_csf ;
}
void get_mode_1_ptr(uint64_t** mode_1_ptr_ttmc, int* size_mode_1_ptr){
    (*mode_1_ptr_ttmc) = mode_1_pointer_csf ;
    *size_mode_1_ptr = size_mode_1_pointer_csf;
}
void get_mode_1_idx(uint64_t** mode_1_idx_ttmc, int* size_mode_1_idx){
    (*mode_1_idx_ttmc) = mode_1_indices_csf ;
    *size_mode_1_idx = size_mode_1_indices_csf ;
}
void get_mode_2_ptr(uint64_t** mode_2_ptr_ttmc, int* size_mode_2_ptr){
    (*mode_2_ptr_ttmc) = mode_2_pointer_csf ;
    *size_mode_2_ptr = size_mode_2_pointer_csf;
}
void get_mode_2_idx(uint64_t** mode_2_idx_ttmc, int* size_mode_2_idx){
    (*mode_2_idx_ttmc) = mode_2_indices_csf ;
    *size_mode_2_idx = size_mode_2_indices_csf ;
}


// Structure to hold the COO data
struct COOElement {
    vector<uint64_t> indices; // Store indices for all modes
    double value;
};


// Function to convert COO to CSF format dynamically
void cooToCSF(uint64_t* my_tensor_indices, double* my_tensor_values, int order, uint64_t  total_indices, uint64_t  total_values) {

    vector<COOElement> cooData;
    vector<uint64_t> indices;


    uint64_t count_idx = 0, count_val = 0;

    for(uint64_t i = 0; i < total_indices; i++){
        
        uint64_t index = my_tensor_indices[count_idx];
        count_idx++;
        indices.push_back(index);
        
        if((i+1)%order == 0){
            double value = my_tensor_values[count_val];
            count_val++;
            cooData.push_back({indices, value});
            indices.clear();
        }

    }




    // uint64_t nnz = cooData.size();
    
    // CSF structure
    vector<vector<uint64_t>> idx(order); // `order` levels of indices
    vector<vector<uint64_t>> ptr(order); // `order` levels of pointers
    vector<uint64_t> vals;   // Store non-zero values
    
    // Initialize mode_0_ptr array with '0'
    ptr[0].push_back(0); 
    
    
    // Sorting the COO data lexicographically by all indices
    vector<COOElement> sortedCOO = cooData;
    sort(sortedCOO.begin(), sortedCOO.end(), [&](const COOElement& a, const COOElement& b) {
        for (int i = 0; i < order; ++i) {
            if (a.indices[i] != b.indices[i]) {
                return a.indices[i] < b.indices[i];
            }
        }
        return false;
    });
    
    // Variables to track the previous indices at each level
    vector<uint64_t> prevIndices(order, -1);
    
    // Loop over the sorted COO data
    for (const auto& elem : sortedCOO) {
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
    for (int level = 0; level < order; ++level) {
        ptr[level].push_back(idx[level].size());  // Final entry to mark the end
    }

    // Output CSF data dynamically based on the detected order
    for (int level = 0; level < order; ++level) {
        if(level == 0){
            size_mode_0_pointer_csf = ptr[level].size();
            mode_0_pointer_csf = new uint64_t [size_mode_0_pointer_csf];
            
            size_mode_0_indices_csf = idx[level].size();
            mode_0_indices_csf = new uint64_t [size_mode_0_indices_csf];

            uint64_t i = 0;
            for (auto val : ptr[level]) {
                mode_0_pointer_csf[i] = val;
                i++;
            }
            
            i = 0;
            for (auto val : idx[level]) {
                mode_0_indices_csf[i] = val;
                i++;
            }            
        }
        else if(level == 1){
            size_mode_1_pointer_csf = ptr[level].size();
            mode_1_pointer_csf = new uint64_t [size_mode_1_pointer_csf];
            
            size_mode_1_indices_csf = idx[level].size();
            mode_1_indices_csf = new uint64_t [size_mode_1_indices_csf];
            
            uint64_t i = 0;
            for (auto val : ptr[level]) {
                mode_1_pointer_csf[i] = val;
                i++;
            }

            i = 0;
            for (auto val : idx[level]) {
                mode_1_indices_csf[i] = val;
                i++;
            }
        }
        else if(level == 2){
            size_mode_2_pointer_csf = ptr[level].size();
            mode_2_pointer_csf = new uint64_t [size_mode_2_pointer_csf];
            
            size_mode_2_indices_csf = idx[level].size();
            mode_2_indices_csf = new uint64_t [size_mode_2_indices_csf];

            uint64_t i = 0;
            for (auto val : ptr[level]) {
                mode_2_pointer_csf[i] = val;
                i++;
            }

            i = 0;
            for (auto val : idx[level]) {
                mode_2_indices_csf[i] = val;
                i++;
            }
        }
    }
}


