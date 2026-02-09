#ifndef CSF_TENSOR_H
#define CSF_TENSOR_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

using namespace std;

// Struct to hold CSF tensor data
struct CSFTensor {
    int order;  // Number of modes
    vector<vector<uint64_t>> ptrs;  // Pointers for each mode
    vector<vector<uint64_t>> idxs;  // Indices for each mode
    vector<float> values;  // Non-zero values
    vector<uint64_t> dimensions; // Dimensions of each mode
    vector<int> modeOrdering; // Original mode ordering
    
    // Initialize with given order
    CSFTensor(int n) : order(n), ptrs(n), idxs(n), dimensions(n), modeOrdering(n) {
        for (int i = 0; i < n; i++) {
            modeOrdering[i] = i; // Default ordering
        }
    }
};

// Function to read a CSF tensor from file
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
            } else if (label.find("_idx") != string::npos) {
                // Parse mode indices
                uint64_t val;
                while (iss >> val) {
                    tensor->idxs[modeIdx].push_back(val);
                }
            }
        } else if (label == "values") {
            // Parse values
            float val;
            while (iss >> val) {
                tensor->values.push_back(val);
            }
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
// Generic, zero-copy version for N-dimensional tensors
inline void getCSFArrays(const CSFTensor& tensor,
                        std::vector<uint64_t*>& mode_ptrs,
                        std::vector<uint64_t*>& mode_idxs,
                        float*& values,
                        int& order) {
    order = tensor.order;
    mode_ptrs.resize(order);
    mode_idxs.resize(order);
    for (int i = 0; i < order; ++i) {
        mode_ptrs[i] = const_cast<uint64_t*>(tensor.ptrs[i].data());
        mode_idxs[i] = const_cast<uint64_t*>(tensor.idxs[i].data());
    }
    values = const_cast<float*>(tensor.values.data());
}


// Helper functions to calculate matrix dimensions based on contraction mode
uint64_t getMatrixDim1(const vector<uint64_t>& dimensions, int contraction_mode) {
    switch (contraction_mode) {
        case 0: return dimensions[1];  // j
        case 1: return dimensions[0];  // i
        case 2: return dimensions[0];  // i
        default: return 0;
    }
}

uint64_t getMatrixDim2(const vector<uint64_t>& dimensions, int contraction_mode) {
    switch (contraction_mode) {
        case 0: return dimensions[2];  // k
        case 1: return dimensions[2];  // k
        case 2: return dimensions[1];  // j
        default: return 0;
    }
}

uint64_t getOutputDim1(const vector<uint64_t>& dimensions, int contraction_mode) {
    switch (contraction_mode) {
        case 0: return dimensions[0];  // i
        case 1: return dimensions[1];  // j
        case 2: return dimensions[2];  // k
        default: return 0;
    }
}

#endif // CSF_TENSOR_H 