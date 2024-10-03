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

// Structure to hold the COO data
struct COOElement {
    vector<int64_t> indices; // Store indices for all modes
    int64_t value;
};

// Function to read COO data from file and detect the order
vector<COOElement> readCOO(const string& filename, int &order) {
    vector<COOElement> cooData;
    ifstream file(filename);
    string line;
    
    while (getline(file, line)) {
        vector<int64_t> indices;
        int64_t value;
        istringstream ss(line);
        int64_t index;

        // Read indices
        while (ss >> index) {
            indices.push_back(index); // Add indices
        }

        // Last element is the value
        value = indices.back();
        indices.pop_back(); // Remove value from indices vector

        // Detect the order based on the number of indices
        if (cooData.empty()) {
            order = indices.size(); // Detect tensor order
        }

        cooData.push_back({indices, value});
    }
    
    file.close();
    return cooData;
}

// Function to convert COO to CSF format dynamically
void cooToCSF(const vector<COOElement>& cooData, int order) {
    int64_t nnz = cooData.size();
    
    // CSF structure
    vector<vector<int64_t>> idx(order); // `order` levels of indices
    vector<vector<int64_t>> ptr(order); // `order` levels of pointers
    vector<int64_t> vals;   // Store non-zero values
    
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
        cout << "mode_" << level << "_ptr : ";
        for (auto val : ptr[level]) {
            cout << val << " ";
        }
        cout << "\nmode_" << level << "_idx : ";
        for (auto val : idx[level]) {
            cout << val << " ";
        }
        cout << "\n\n";
    }

    // Output the values
    cout << "vals : ";
    for (auto val : vals) {
        cout << val << " ";
    }
    cout << endl;
}

int main() {
    // Read the COO data from file and dynamically detect the order
    string filePath = "coo.txt";  // Specify the file path 
    int order = 0;
    vector<COOElement> cooData = readCOO(filePath, order);
    
    // Convert to CSF format
    cooToCSF(cooData, order);

    return 0;
}
