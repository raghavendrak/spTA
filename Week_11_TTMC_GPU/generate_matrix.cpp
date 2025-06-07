#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <cstdint>

// Function to read COO file and get dimensions
bool readCOOFile(const std::string& filename, int& tensorOrder, std::vector<uint64_t>& dimensions) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return false;
    }

    // Read tensor order
    file >> tensorOrder;

    // Read dimensions
    dimensions.resize(tensorOrder);
    for (int i = 0; i < tensorOrder; ++i) {
        file >> dimensions[i];
    }

    file.close();
    return true;
}

// Function for writing a matrix to file
void writeMatrixToFile(const std::string& filename, uint64_t rows, uint64_t cols, unsigned int seed) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    std::mt19937 gen(seed); // Mersenne Twister RNG seeded with 'seed'
    std::uniform_real_distribution<double> dist(0.0, 1.0); // Uniform distribution in [0, 1)

    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            file << dist(gen) << " ";
        }
        file << "\n";
    }

    file.close();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <coo_file> [rank1] [rank2] [rank3]" << std::endl;
        return 1;
    }

    std::string cooFileName = argv[1];
    
    // Default ranks
    std::vector<int> ranks = {30, 30, 30};
    
    // Read ranks from command line if provided
    for (int i = 0; i < 3 && i + 2 < argc; ++i) {
        ranks[i] = std::stoi(argv[i + 2]);
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

    // Generate random seed
    unsigned int seed = static_cast<unsigned int>(time(nullptr));

    // Generate 3 matrices
    for (int i = 0; i < 3; ++i) {
        if (i < tensorOrder) {
            uint64_t rows = dimensions[i];
            uint64_t cols = ranks[i];
            
            // Generate output filename based on COO file
            std::string baseName = cooFileName;
            // Remove extension if present
            size_t lastDotPos = baseName.find_last_of('.');
            if (lastDotPos != std::string::npos) {
                baseName = baseName.substr(0, lastDotPos);
            }
            std::string outputFile = baseName + "_dim" + std::to_string(i) + ".tns";
            
            // Generate random matrix and write to file
            std::cout << "Generating matrix " << i << ": " << rows << "x" << cols << " matrix..." << std::endl;
            writeMatrixToFile(outputFile, rows, cols, seed + i); // Use different seed for each matrix
            
            std::cout << "Matrix " << i << " written to " << outputFile << std::endl;
        } else {
            std::cout << "Skipping matrix " << i << " as tensor order is " << tensorOrder << std::endl;
        }
    }
    
    return 0;
} 