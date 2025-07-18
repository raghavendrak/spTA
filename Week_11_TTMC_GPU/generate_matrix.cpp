#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <cstdint>
#include <cstring>
#include <sstream>

// Function to read COO file and get dimensions
bool readCOOFile(const std::string& filename, int& tensorOrder, std::vector<uint64_t>& dimensions) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return false;
    }

    try {
        // Read tensor order
        std::string line;
        std::getline(file, line);
        tensorOrder = std::stoi(line);

        // Read dimensions
        std::getline(file, line);
        std::istringstream iss(line);
        dimensions.clear();
        dimensions.resize(tensorOrder);
        
        for (int i = 0; i < tensorOrder; ++i) {
            if (!(iss >> dimensions[i])) {
                std::cerr << "Error: Failed to read dimension " << i << std::endl;
                file.close();
                return false;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing COO file: " << e.what() << std::endl;
        file.close();
        return false;
    }

    file.close();
    return true;
}

// Function for writing a matrix to file in dense format
void writeMatrixToDenseFile(const std::string& filename, uint64_t rows, uint64_t cols, unsigned int seed) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    std::mt19937 gen(seed); // Mersenne Twister RNG seeded with 'seed'
    std::uniform_real_distribution<float> dist(0.0, 1.0); // Uniform distribution in [0, 1)

    // Write dimensions first (compatible with ParTI's Tensor::load)
    file << "2\n"; // 2D tensor (matrix)
    file << rows << " " << cols << "\n";

    // Write values
    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            file << dist(gen) << " ";
        }
        file << "\n";
    }

    file.close();
}

// Function for writing a matrix to file in COO format
void writeMatrixToCOOFile(const std::string& filename, uint64_t rows, uint64_t cols, unsigned int seed) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    std::mt19937 gen(seed); // Mersenne Twister RNG seeded with 'seed'
    std::uniform_real_distribution<float> dist(0.0, 1.0); // Uniform distribution in [0, 1)

    // Write dimensions first (compatible with ParTI's Tensor::load)
    file << "2\n"; // 2D tensor (matrix)
    file << rows << " " << cols << "\n";

    // Generate and write non-zero values in COO format
    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            float val = dist(gen);
            file << i << " " << j << " " << val << "\n";
        }
    }

    file.close();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <coo_file> [rank1] [rank2] [rank3] [--coo]" << std::endl;
        std::cerr << "  --coo  : Generate matrices in COO format (default is dense format)" << std::endl;
        return 1;
    }

    std::string cooFileName = argv[1];
    
    // Default ranks
    std::vector<int> ranks = {30, 30, 30};
    
    // Check for format flag
    bool useCOOFormat = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--coo") == 0) {
            useCOOFormat = true;
        }
    }
    
    // Process non-flag arguments for ranks
    std::vector<std::string> nonFlagArgs;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') {
            nonFlagArgs.push_back(argv[i]);
        }
    }
    
    // First non-flag arg is the filename, rest are ranks
    for (size_t i = 1; i < nonFlagArgs.size() && i <= 3; ++i) {
        try {
            ranks[i-1] = std::stoi(nonFlagArgs[i]);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Invalid rank argument '" << nonFlagArgs[i] 
                      << "', using default rank: " << e.what() << std::endl;
        }
    }

    // Read COO file
    int tensorOrder;
    std::vector<uint64_t> dimensions;
    if (!readCOOFile(cooFileName, tensorOrder, dimensions)) {
        return 1;
    }

    // Display information
    std::cout << "Tensor Order: " << tensorOrder << std::endl;
    std::cout << "Dimensions: ";
    for (int i = 0; i < tensorOrder; ++i) {
        std::cout << dimensions[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Ranks: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << ranks[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Format: " << (useCOOFormat ? "COO" : "Dense") << std::endl;

    // Generate random seed
    unsigned int seed = static_cast<unsigned int>(time(nullptr));

    // Generate matrices for TTM operation
    // For TTM, each matrix should have dimensions [R × I_n]
    // where I_n is the size of the tensor in mode n, and R is the rank
    for (int i = 1; i < tensorOrder && i < 3; ++i) {
        // For TTM, Matrix U should have dimensions [R × I_n]
        // where R is the rank (nrows) and I_n is the dimension of mode n (ncols)
        uint64_t rows = ranks[i]; // Rank for this mode
        uint64_t cols = dimensions[i]; // Dimension of tensor in this mode
        
        std::cout << "For TTM operation on mode " << i << ", matrix must have columns matching tensor dimension " << cols << std::endl;
        
        // Generate output filename based on COO file
        std::string baseName = cooFileName;
        // Remove extension if present
        size_t lastDotPos = baseName.find_last_of('.');
        if (lastDotPos != std::string::npos) {
            baseName = baseName.substr(0, lastDotPos);
        }
        std::string outputFile = baseName + "_dim" + std::to_string(i) + ".tns";
        
        // Generate random matrix and write to file in the selected format
        std::cout << "Generating matrix " << i << ": " << rows << "x" << cols << " matrix..." << std::endl;
        
        if (useCOOFormat) {
            writeMatrixToCOOFile(outputFile, rows, cols, seed + i);
        } else {
            writeMatrixToDenseFile(outputFile, rows, cols, seed + i);
        }
        
        std::cout << "Matrix " << i << " written to " << outputFile << std::endl;
    }
    
    return 0;
} 