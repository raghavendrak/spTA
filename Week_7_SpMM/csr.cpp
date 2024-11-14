#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <sstream>
#include <algorithm> 

struct CSR {
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_pointers;
};

// Function to read Matrix Market file and convert to CSR
CSR convertToCSR(const std::string &filename) {
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

    // Temporary storage for entries
    std::vector<std::tuple<int, int, double>> entries;
    int row, col;
    double value;

    // Read non-zero entries
    while (file >> row >> col >> value) {
        entries.emplace_back(row - 1, col - 1, value); // Convert to 0-based indexing
        // entries.emplace_back(row , col , value);
    }

    // Sort entries by row and column if necessary (optional)
    std::sort(entries.begin(), entries.end());

    // Initialize CSR structure
    CSR csr;
    csr.values.reserve(non_zeros);
    csr.col_indices.reserve(non_zeros);
    csr.row_pointers.resize(rows + 1, 0);

    // Fill CSR arrays
    int current_row = 0;
    for (const auto &[r, c, v] : entries) {
        while (current_row < r) {
            csr.row_pointers[++current_row] = csr.values.size();
        }
        csr.values.push_back(v);
        csr.col_indices.push_back(c);
    }
    csr.row_pointers[rows] = csr.values.size();

    return csr;
}

// Function to display the CSR matrix
void displayCSR(const CSR &csr) {
    std::cout << "Values: ";
    for (double v : csr.values) std::cout << v << " ";
    std::cout << "\nColumn Indices: ";
    for (int col : csr.col_indices) std::cout << col << " ";
    std::cout << "\nRow Pointers: ";
    for (int ptr : csr.row_pointers) std::cout << ptr << " ";
    std::cout << std::endl;
}

int main() {
    std::string filename = "mhda416.mtx"; // replace with your file name
    try {
        CSR csr = convertToCSR(filename);
        displayCSR(csr);
    } catch (const std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
