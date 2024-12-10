#ifndef GENTEN_H
#define GENTEN_H

#ifdef __cplusplus
extern "C" {
#endif

// Declare the functions you need to use in the CUDA file
void generate_tensor(int argc, char** argv, long** my_tensor_indices, double** my_tensor_values, int* total_indices, int* total_values);

#ifdef __cplusplus
}
#endif

#endif // GENTEN_H
