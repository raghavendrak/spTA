#!/bin/bash

# Script to benchmark TTM operations on all tensors
# This script generates matrices, runs TTM chain tests, and extracts kernel times
# Usage: ./run_ttm_benchmark.sh <r1> <r2>

# Check command line arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <r1> <r2>"
    echo "Example: $0 10 10"
    exit 1
fi

R1="$1"
R2="$2"

echo "Running TTM benchmark with r1=$R1, r2=$R2"

TENSOR_DIR="/home/bhaskar/tensors_dataset"
GENERATE_MATRIX="/home/bhaskar/spTA/Week_11_TTMC_GPU/generate_matrix.out"
TTM_TEST="/home/bhaskar/ParTI/build/tests/test_ttm_chain"
LOG_FILE="/tmp/ttm_benchmark.log"
RESULTS_FILE="/tmp/ttm_results.txt"

# Clear previous results
> "$RESULTS_FILE"
> "$LOG_FILE"

echo "Starting TTM benchmark for all tensors with r1=$R1, r2=$R2..."
echo "Dataset,Total_CUDA_TTM_Kernel_Time(s)" > "$RESULTS_FILE"

# Function to extract TTM kernel times from log output
extract_ttm_times() {
    local log_content="$1"
    local total_time=0
    
    # Extract all CUDA TTM Kernel times using regex
    while IFS= read -r line; do
        if [[ "$line" =~ \[CUDA[[:space:]]+TTM[[:space:]]+Kernel\]:[[:space:]]+([0-9]+\.[0-9]+)[[:space:]]+s[[:space:]]+spent ]]; then
            time_value="${BASH_REMATCH[1]}"
            total_time=$(echo "$total_time + $time_value" | bc -l)
        fi
    done <<< "$log_content"
    
    echo "$total_time"
}

# Function to clean up generated matrices
cleanup_matrices() {
    local base_name="$1"
    rm -f "${TENSOR_DIR}/${base_name}_dim1.tns" "${TENSOR_DIR}/${base_name}_dim2.tns" 2>/dev/null
}

# Process each tensor file
for tensor_file in "$TENSOR_DIR"/*.tns; do
    if [[ ! -f "$tensor_file" ]]; then
        continue
    fi
    
    base_name=$(basename "$tensor_file" .tns)
    echo "Processing $base_name..."
    
    # Skip if no corresponding CSF file exists (like matmul tensors)
    csf_file="/home/bhaskar/spTA/Week_11_TTMC_GPU/${base_name}.csf"
    if [[ ! -f "$csf_file" ]]; then
        echo "  Skipping $base_name: No corresponding CSF file"
        continue
    fi
    
    # Generate matrices
    echo "  Generating matrices for $base_name with r1=$R1, r2=$R2..."
    matrix_output=$("$GENERATE_MATRIX" "$tensor_file" "$R1" "$R2" 2>&1)
    if [[ $? -ne 0 ]]; then
        echo "  Error generating matrices for $base_name"
        continue
    fi
    
    # Check if matrices were generated
    dim1_matrix="${TENSOR_DIR}/${base_name}_dim1.tns"
    dim2_matrix="${TENSOR_DIR}/${base_name}_dim2.tns"
    
    if [[ ! -f "$dim1_matrix" || ! -f "$dim2_matrix" ]]; then
        echo "  Error: Matrices not generated for $base_name"
        continue
    fi
    
    # Run TTM chain test
    echo "  Running TTM chain test for $base_name..."
    ttm_output=$("$TTM_TEST" --dev 48 "$tensor_file" "$dim2_matrix" "$dim1_matrix" -l 2 2>&1)
    ttm_exit_code=$?
    
    # Log the output for debugging
    echo "=== $base_name ===" >> "$LOG_FILE"
    echo "$ttm_output" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    if [[ $ttm_exit_code -eq 0 ]]; then
        # Extract TTM kernel times
        total_time=$(extract_ttm_times "$ttm_output")
        echo "  Total TTM kernel time for $base_name: ${total_time}s"
        echo "$base_name,$total_time" >> "$RESULTS_FILE"
    else
        echo "  Error running TTM test for $base_name"
        echo "$base_name,ERROR" >> "$RESULTS_FILE"
    fi
    
    # Clean up generated matrices
    cleanup_matrices "$base_name"
    
    echo "  Completed $base_name"
    echo ""
done

echo "Benchmark completed. Results saved to $RESULTS_FILE"
echo "Detailed logs saved to $LOG_FILE"

# Display results table using Python script if available
if [[ -f "/home/bhaskar/spTA/Week_11_TTMC_GPU/extract_ttm_times.py" ]]; then
    echo ""
    echo "=== TTM Benchmark Results ==="
    python3 /home/bhaskar/spTA/Week_11_TTMC_GPU/extract_ttm_times.py
else
    # Fallback to simple column display
    echo ""
    echo "=== TTM Benchmark Results ==="
    column -t -s',' "$RESULTS_FILE"
fi
