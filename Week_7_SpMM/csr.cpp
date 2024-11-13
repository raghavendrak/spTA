#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <sstream>
#include <algorithm>
#include "csr_data.h"

// Function to read Matrix Market file and convert to CSR
void convertToCSR(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file");
    }

    std::string line;
    int rows, cols, non_zeros;

    // Skip header lines
    while (std::getline(file, line)) {
        if (line[0] != '%') {
            break;
        }
    }

    // Read matrix dimensions and number of non-zero entries
    std::istringstream iss(line);
    iss >> rows >> cols >> non_zeros;

    // Assign global variables
    A_rows = rows;
    A_cols = cols;
    A_nonzeros = non_zeros;

    // Temporary storage for entries
    std::vector<std::tuple<int, int, double>> entries;
    int row, col;
    double value;

    // Read non-zero entries
    while (file >> row >> col >> value) {
        entries.emplace_back(row - 1, col - 1, value); // Convert to 0-based indexing
    }

    // Sort entries by row and column if necessary (optional)
    std::sort(entries.begin(), entries.end());

    // Resize global vectors
    values.reserve(non_zeros);
    col_indices.reserve(non_zeros);
    row_pointers.resize(rows + 1, 0);

    // Fill CSR arrays
    int current_row = 0;
    for (const auto &[r, c, v] : entries) {
        while (current_row < r) {
            row_pointers[++current_row] = values.size();
        }
        values.push_back(v);
        col_indices.push_back(c);
    }
    row_pointers[rows] = values.size();
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <MatrixMarketFile>" << std::endl;
        return 1;
    }

    try {
        std::string filename = argv[1];
        convertToCSR(filename);
        displayCSR(); // Display the CSR matrix (for verification)
    } catch (const std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
