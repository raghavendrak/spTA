#!/bin/bash

# Compile the program if not already compiled
nvcc -o gemm_cuda_cublas gemm_cuda_cublas.cu -lcublas

# Initialize variables for dimensions and number of runs
M=30000
N=30000
K=30000
NUM_RUNS=5
TOTAL_TIME=0

# Create or clear the results file
echo "Run,Execution Time (seconds)" > results_cublas.csv

# Run the program multiple times and collect execution time
for ((i=1; i<=NUM_RUNS; i++))
do
    # Capture the execution time for each run
    EXEC_TIME=$(./gemm_cuda_cublas $M $N $K | grep "GPU execution time using cuBLASD" | awk '{print $(NF-1)}')
    
    # Append execution time to CSV
    echo "$i,$EXEC_TIME" >> results_cublas.csv
    
    # Add to total time for averaging
    TOTAL_TIME=$(echo "$TOTAL_TIME + $EXEC_TIME" | bc)
done

# Calculate the average execution time
AVERAGE_TIME=$(echo "scale=6; $TOTAL_TIME / $NUM_RUNS" | bc)

# Append the average to the CSV file
echo "Average,$AVERAGE_TIME" >> results_cublas.csv

# Print message indicating completion
echo "Execution completed. Results saved in results_cublas.csv"
