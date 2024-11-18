#!/bin/bash

# Compile the program
nvcc -o spMM_cuda spMM_cuda.cu COOtoCSR.cpp

# Define the matrix file and number of columns in Matrix B
MATRIX_FILE="bloweybl.mtx"
NUM_COLUMNS="30003"

# Thread block sizes to test
THREAD_BLOCK_SIZES=(32 64 128 256 512 1024)

# Output file for results
RESULT_FILE="results.csv"

# Initialize the results.csv file
echo "Thread Block Size,Average Execution Time (seconds)" > $RESULT_FILE

# Function to run the program 10 times and calculate the average execution time
run_test() {
    local block_size=$1
    local total_time=0

    echo "Running test for thread block size: $block_size"
    
    for i in {1..5}; do
        # Run the program and capture the execution time
        output=$(./spMM_cuda $MATRIX_FILE $NUM_COLUMNS $block_size)
        
        # Extract the execution time from the output
        time=$(echo "$output" | grep -oP 'SpMM GPU execution time: \K[0-9]+\.[0-9]+')
        echo "Run $i: $time seconds"
        
        # Accumulate the time
        total_time=$(echo "$total_time + $time" | bc)
    done

    # Calculate the average time
    avg_time=$(echo "scale=9; $total_time / 10" | bc)
    echo "Average execution time for thread block size $block_size: $avg_time seconds"
    echo

    # Save the results to the results.csv file
    echo "$block_size,$avg_time" >> $RESULT_FILE
}

# Loop over the defined thread block sizes
for block_size in "${THREAD_BLOCK_SIZES[@]}"; do
    run_test $block_size
done

echo "Results have been saved to $RESULT_FILE"
