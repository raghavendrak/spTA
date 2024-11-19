#ifndef CSR_DATA_H
#define CSR_DATA_H

// Function prototypes
void convertToCSR(const std::string &filename, int64_t*& row_pointers, int64_t*& col_indices, double*& values, int64_t& A_rows, int64_t& A_cols, int64_t& A_nonzeros);

#endif // CSR_DATA_H
