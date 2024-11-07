#!/bin/bash

# Compile the CUDA program
nvcc -o gemm_cuda gemm_cuda.cu

# Matrix dimensions
M=30000
N=30000
K=30000

# Output CSV file
OUTPUT_FILE="results.csv"
echo "Thread Block Size,Average Execution Time (seconds)" > $OUTPUT_FILE

# Run for different thread block sizes and save the average results
for block_size in 8 16 32 64; do
    total_time=0
    
    # Run the program 10 times and sum the execution times
    for i in {1..5}; do
        # Run the program and capture the output
        output=$(./gemm_cuda $M $N $K $block_size)
        
        # Extract the execution time from the output
        exec_time=$(echo "$output" | grep "GPU execution time" | awk '{print $(NF-1)}')
        echo "For Thread Block Size : $block_size, Run $i : $exec_time sec"
        
        # Add the execution time to the total
        total_time=$(echo "$total_time + $exec_time" | bc)
    done
    
    # Calculate the average execution time
    avg_time=$(echo "scale=2; $total_time / 5" | bc)
    
    # Save the block size and average execution time to the CSV file
    echo "$block_size,$avg_time" >> $OUTPUT_FILE
done

echo "Results saved to $OUTPUT_FILE"
