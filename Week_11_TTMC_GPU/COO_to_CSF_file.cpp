#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <sstream>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <omp.h>

using namespace std;

/**
 * COO_to_CSF_file.cpp
 * This program reads a tensor in COO format from a .tns file and converts it
 * to CSF format, writing the result to a file with each level of pointers and indices
 * on separate lines.
 * 
 * The program implements SPLATT's optimization: reordering tensor modes by increasing
 * length (number of unique indices) to maximize the branching factor of the CSF tree.
 * 
 * Updated format:
 * - First non-empty, non-comment line: order of the tensor
 * - Second non-empty, non-comment line: dimensions of each mode
 * - Remaining lines: COO format with indices and values
 * 
 * Usage: ./COO_to_CSF_file <input_file.tns> <output_file.csf> [num_modes]
 * If num_modes is not specified, the program will detect it from the input file.
 */

// Structure to hold the COO data
struct COOElement {
    vector<uint64_t> indices; // Store indices for all modes
    double value;
};

// Function to parse the COO file and convert to CSF format
void cooToCSF(const string& inputFile, const string& outputFile, int specifiedOrder = 0) {
    vector<COOElement> cooData;
    ifstream inFile(inputFile);
    if (!inFile) {
        cerr << "Error: Unable to open input file: " << inputFile << endl;
        exit(1);
    }

    string line;
    int order = specifiedOrder;
    vector<uint64_t> dimensions;
    bool orderRead = false;
    bool dimensionsRead = false;
    
    // Read the COO data, with special handling for first two non-empty lines
    while (getline(inFile, line)) {
        // Skip empty lines or comments
        if (line.empty() || line[0] == '#') continue;
        
        // First non-empty line: read the order
        if (!orderRead) {
            istringstream iss(line);
            if (specifiedOrder == 0) {
                iss >> order;
                cout << "Read tensor order: " << order << endl;
            }
            orderRead = true;
            continue;
        }
        
        // Second non-empty line: read the dimensions
        if (!dimensionsRead) {
            istringstream iss(line);
            uint64_t dim;
            while (iss >> dim) {
                dimensions.push_back(dim);
            }
            
            if (dimensions.size() != order) {
                cerr << "Error: Number of dimensions (" << dimensions.size() 
                     << ") doesn't match order (" << order << ")" << endl;
                exit(1);
            }
            
            cout << "Read tensor dimensions: ";
            for (uint64_t dim : dimensions) {
                cout << dim << " ";
            }
            cout << endl;
            
            dimensionsRead = true;
            continue;
        }
        
        // Remaining lines: read COO data
        istringstream iss(line);
        vector<uint64_t> indices;
        uint64_t idx;
        double value;
        
        // Read the indices
        int i = 0;
        while (i < order && (iss >> idx)) {
            indices.push_back(idx);
            i++;
        }
        
        // Read the value
        if (!(iss >> value)) {
            cerr << "Error: Missing value in line: " << line << endl;
            exit(1);
        }
        
        // Ensure we have the correct number of indices
        if (indices.size() != order) {
            cerr << "Error: Inconsistent number of indices in line: " << line << endl;
            exit(1);
        }
        
        cooData.push_back({indices, value});
    }
    
    inFile.close();

    // Find the number of unique indices for each mode
    vector<unordered_set<uint64_t>> uniqueIndices(order);
    for (const auto& elem : cooData) {
        for (int i = 0; i < order; i++) {
            uniqueIndices[i].insert(elem.indices[i]);
        }
    }

    // Create a vector of mode indices sorted by the number of unique indices (ascending)
    vector<int> modeOrder(order);
    for (int i = 0; i < order; i++) {
        modeOrder[i] = i;
    }
    sort(modeOrder.begin(), modeOrder.end(),
         [&uniqueIndices](int a, int b) {
             return uniqueIndices[a].size() < uniqueIndices[b].size();
         });
    
    // Print the mode ordering for diagnostic purposes
    cout << "Mode ordering (smallest to largest): ";
    for (int mode : modeOrder) {
        cout << mode << " (" << uniqueIndices[mode].size() << " unique indices) ";
    }
    cout << endl;

    // Reorder the dimensions according to the new mode ordering
    vector<uint64_t> reorderedDimensions(order);
    for (int i = 0; i < order; i++) {
        reorderedDimensions[i] = dimensions[modeOrder[i]];
    }

    // Reorder the indices in each COO element according to the new mode ordering
    #pragma omp parallel for
    for (auto& elem : cooData) {
        vector<uint64_t> reorderedIndices(order);
        for (int i = 0; i < order; i++) {
            reorderedIndices[i] = elem.indices[modeOrder[i]];
        }
        elem.indices = reorderedIndices;
    }
    
    // Sort the COO data lexicographically with the new ordering
    sort(cooData.begin(), cooData.end(), [](const COOElement& a, const COOElement& b) {
        return a.indices < b.indices;
    });
    
    // Generate CSF format
    vector<vector<uint64_t>> idx(order); // Indices for each level
    vector<vector<uint64_t>> ptr(order); // Pointers for each level
    vector<double> vals; // Non-zero values
    
    // Initialize the first level pointer
    ptr[0].push_back(0);
    
    // Variables to track the previous indices at each level
    vector<uint64_t> prevIndices(order, UINT64_MAX);
    
    // Convert to CSF
    for (const auto& elem : cooData) {
        bool fiberChanged = false;
        
        for (int level = 0; level < order; level++) {
            if (elem.indices[level] != prevIndices[level] || fiberChanged) {
                fiberChanged = true;
                
                // New index for this level
                idx[level].push_back(elem.indices[level]);
                
                // Update pointer for the next level
                if (level + 1 < order) {
                    ptr[level + 1].push_back(idx[level + 1].size());
                }
                
                // Update previous index
                prevIndices[level] = elem.indices[level];
            }
        }
        
        // Store the value
        vals.push_back(elem.value);
    }
    
    // Finalize pointers for all levels
    for (int level = 0; level < order; level++) {
        ptr[level].push_back(idx[level].size());
    }
    
    // Write the CSF data to file
    ofstream outFile(outputFile);
    if (!outFile) {
        cerr << "Error: Unable to open output file: " << outputFile << endl;
        exit(1);
    }
    
    // Write the order as a comment
    outFile << "# Tensor order: " << order << endl;
    
    // Write the original mode ordering
    outFile << "# Mode ordering:";
    for (int i = 0; i < order; i++) {
        outFile << " " << modeOrder[i];
    }
    outFile << endl;
    
    // Write the reordered dimensions
    outFile << "# Dimensions:";
    for (uint64_t dim : reorderedDimensions) {
        outFile << " " << dim;
    }
    outFile << endl;
    
    // Write pointers and indices for each mode
    for (int level = 0; level < order; level++) {
        // Write pointers
        outFile << "mode_" << level << "_ptr:";
        for (uint64_t val : ptr[level]) {
            outFile << " " << val;
        }
        outFile << endl;
        
        // Write indices
        outFile << "mode_" << level << "_idx:";
        for (uint64_t val : idx[level]) {
            outFile << " " << val - 1; // Convert to 0-based indexing since input is 1-based
        }
        outFile << endl;
    }
    
    // Write values
    outFile << "values:";
    for (double val : vals) {
        outFile << " " << val;
    }
    outFile << endl;
    
    outFile.close();
    
    // Print summary
    cout << "Conversion complete!" << endl;
    cout << "Number of non-zero elements: " << vals.size() << endl;
    for (int level = 0; level < order; level++) {
        cout << "Mode " << level << " (original mode " << modeOrder[level] << ") size: " << idx[level].size() << endl;
        cout << "  Dimension: " << reorderedDimensions[level] << endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_file.tns> <output_file.csf> [num_modes]" << endl;
        return 1;
    }
    
    string inputFile = argv[1];
    string outputFile = argv[2];
    int order = 0;  // Auto-detect by default
    
    // If order is specified as an argument
    if (argc > 3) {
        order = atoi(argv[3]);
    }
    
    try {
        cooToCSF(inputFile, outputFile, order);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
} 