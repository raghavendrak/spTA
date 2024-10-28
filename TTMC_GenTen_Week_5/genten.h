#ifndef GENTEN_H
#define GENTEN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>

#define ULLI unsigned long long int
#define USHI unsigned short int 
#define INTSIZE 268435456
#define APPLY_IMBALANCE 1
#define RATIO_MIN 0.95
#define RATIO_MAX 1.05

// Function prototypes
void *safe_malloc(size_t size);
void *safe_calloc(size_t count, size_t size);
void printusage();
double norm_box_muller(double mean, double stdev, int seed_bm);
double calculate_std(int *arr, int arr_size, double mean);
void print_vec(ULLI *array, int array_size);
void print_vec_double(double *array, int array_size);

// Tensor generation function prototype
void generate_tensor(int argc, char *argv[], int64_t **my_tensor_indices, double **my_tensor_values, int64_t  *total_indices, int64_t  *total_values);

#endif // GENTEN_H
