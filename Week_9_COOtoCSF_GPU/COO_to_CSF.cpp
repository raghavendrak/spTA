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
#include "genten.h"
#include <chrono>

using namespace std;

// Structure to hold the COO data
struct COOElement {
    vector<int64_t> indices; // Store indices for all modes
    double value;
};


// Function to convert COO to CSF format dynamically
void cooToCSF(const vector<COOElement>& cooData, int order) {
    int64_t nnz = cooData.size();
    
    // CSF structure
    vector<vector<int64_t>> idx(order); // `order` levels of indices
    vector<vector<int64_t>> ptr(order); // `order` levels of pointers
    vector<double> vals;   // Store non-zero values
    
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
    vector<int64_t> prevIndices(order, -1);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    
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
    for (int level = 0; level < order; ++level){
        ptr[level].push_back(idx[level].size());  // Final entry to mark the end
    }

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    std::cout << "COO to CSF CPU execution time: " << cpu_time.count() << " seconds." << std::endl;

    // Output CSF data dynamically based on the detected order
    // for (int level = 0; level + 1 < order; ++level) {
    //     cout << "mode_" << level << "_ptr : ";
    //     for (auto val : ptr[level]) {
    //         cout << val << " ";
    //     }
    //     cout << "\nmode_" << level << "_idx : ";
    //     for (auto val : idx[level]) {
    //         cout << val << " ";
    //     }
    //     cout << "\n\n";
    // }

    // Output the values
    // cout << "vals : ";
    // for (auto val : vals) {
    //     cout << val << " ";
    // }
    // cout << endl;
}


/*
NOTE : Make sure to change the input matrix files as per the chosen contraction : 





Command for compiling this program : 
g++ -O2 -Wall -fopenmp  COO_to_CSF.cpp genten.c -o cootocsf


Command to run this program : 
./cootocsf 3 2000 2000 2000 -d 0.01 -f 0.1 -c 0.5 -v 0.5 -o ../sample_data/generated_100_3D.tns

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

    vector<COOElement> cooData(total_values); // Initialize with number of non-zero values
    for(int i=0, j=0; i < total_indices; i++) {
        cooData[j].indices.push_back(my_tensor_indices[i]-1);
        if((i + 1) % order == 0) {
            cooData[j].value = my_tensor_values[j]; // Assign the value after indices
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
    //     if((i+1)%3 == 0){
    //         cout << my_tensor_values[j];
    //         j++;
    //         cout << endl;
    //     }
    // }
    // cout << endl;

    cooToCSF(cooData, order);


    return 0;
}
