#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <sstream>
#include <algorithm>
#include "COOtoCSR.h"

// Function to read Matrix Market file and convert to CSR
void convertToCSR(const std::string &filename, int64_t*& row_pointers, int64_t*& col_indices, double*& values, int64_t& A_rows, int64_t& A_cols, int64_t& A_nonzeros) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file");
    }

    std::string line;

    // Skip header lines
    while (std::getline(file, line)) {
        if (line[0] != '%') {
            break;
        }
    }

    // Read matrix dimensions and number of non-zero entries
    std::istringstream iss(line);
    iss >> A_rows >> A_cols >> A_nonzeros;

    // Temporary storage for entries
    std::vector<std::tuple<int, int, double>> entries;
    int row, col;
    double value;

    // Read non-zero entries
    while (file >> row >> col >> value) {
        entries.emplace_back(row - 1, col - 1, value); // Convert to 0-based indexing
        //entries.emplace_back(row , col , value);
    }

    // Sort entries by row and column if necessary (optional)
    std::sort(entries.begin(), entries.end());

    // Allocate memory for the arrays
    values = new double[A_nonzeros];
    col_indices = new int64_t[A_nonzeros];
    row_pointers = new int64_t[A_rows + 1];

    // Fill CSR arrays
    int current_row = 0;
    int value_index = 0;

    // Initialize row pointers to zero
    for (int i = 0; i <= A_rows; ++i) {
        row_pointers[i] = 0;
    }

    // Fill arrays with the CSR format data
    for (const auto &[r, c, v] : entries) {
        while (current_row < r) {
            row_pointers[++current_row] = value_index;
        }
        values[value_index] = v;
        col_indices[value_index] = c;
        value_index++;
    }
    row_pointers[A_rows] = value_index;
}
