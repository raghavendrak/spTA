#ifndef COO_TO_CSF_H
#define COO_TO_CSF_H

#include <cstdint>

extern int max_size;

// Declare the global variables for CSF structure
extern int64_t* mode_0_pointer_csf;
extern int64_t* mode_0_indices_csf;
extern int64_t* mode_1_pointer_csf;
extern int64_t* mode_1_indices_csf;
extern int64_t* mode_2_pointer_csf;
extern int64_t* mode_2_indices_csf;

extern int size_mode_0_pointer_csf;
extern int size_mode_0_indices_csf;
extern int size_mode_1_pointer_csf;
extern int size_mode_1_indices_csf;
extern int size_mode_2_pointer_csf;
extern int size_mode_2_indices_csf;

// Function declarations to get pointers and indices for each mode
void get_mode_0_ptr(int64_t** mode_0_ptr_ttmc, int* size_mode_0_ptr);
void get_mode_0_idx(int64_t** mode_0_idx_ttmc, int* size_mode_0_idx);
void get_mode_1_ptr(int64_t** mode_1_ptr_ttmc, int* size_mode_1_ptr);
void get_mode_1_idx(int64_t** mode_1_idx_ttmc, int* size_mode_1_idx);
void get_mode_2_ptr(int64_t** mode_2_ptr_ttmc, int* size_mode_2_ptr);
void get_mode_2_idx(int64_t** mode_2_idx_ttmc, int* size_mode_2_idx);

// Structure to hold the COO data
struct COOElement {
    std::vector<int64_t> indices; // Store indices for all modes
    int64_t value;
};

// Function to convert COO format to CSF format
void cooToCSF(int64_t* my_tensor_indices, double* my_tensor_values, int order, int total_indices, int total_values);

#endif // COO_TO_CSF_H
