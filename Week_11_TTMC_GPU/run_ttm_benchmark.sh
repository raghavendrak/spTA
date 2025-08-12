#!/bin/bash

# Script to benchmark TTM operations on all tensors
# This script generates matrices, runs TTM chain tests, and extracts kernel times
# Usage: ./run_ttm_benchmark.sh <r1> <r2> ... <rankN>

# Check command line arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <rank1> <rank2> ... <rankN>"
    echo "Example: $0 10 10"
    exit 1
fi

RANKS=("$@")

echo -n "Running TTM benchmark with ranks:"
for r in "${RANKS[@]}"; do
    echo -n " $r"
done
echo

TENSOR_DIR="/home/bhaskar/tensors_dataset"
GENERATE_MATRIX="/home/bhaskar/spTA/Week_11_TTMC_GPU/generate_matrix.out"
TTM_TEST="/home/bhaskar/ParTI/build/tests/test_ttm_chain"
LOG_FILE="/home/bhaskar/spTA/Week_11_TTMC_GPU/ttm_benchmark.log"
RESULTS_FILE="/home/bhaskar/spTA/Week_11_TTMC_GPU/ttm_results.txt"

# Clear previous results
> "$RESULTS_FILE"
> "$LOG_FILE"

echo -n "Starting TTM benchmark for all tensors with ranks:"
for r in "${RANKS[@]}"; do
    echo -n " $r"
done
echo

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
    echo "  Generating matrices for $base_name with ranks:"
    for r in "${RANKS[@]}"; do
        echo -n " $r"
    done
    echo
    matrix_output=$("$GENERATE_MATRIX" "$tensor_file" "${RANKS[@]}" 2>&1)
    if [[ $? -ne 0 ]]; then
        echo "  Error generating matrices for $base_name"
        continue
    fi
    
    # Check if matrices were generated
    dim1_matrix="${TENSOR_DIR}/${base_name}_dim1.tns"
    dim2_matrix="${TENSOR_DIR}/${base_name}_dim2.tns"
    dim3_matrix="${TENSOR_DIR}/${base_name}_dim3.tns"

    # Read tensor order from first line of tensor_file
    tensor_order=$(head -n 1 "$tensor_file")
    if [[ -z "$tensor_order" ]]; then
        echo "  Error: Could not read tensor order from $tensor_file"
        continue
    fi

    if [[ "$tensor_order" -eq 3 ]]; then
        if [[ ! -f "$dim1_matrix" || ! -f "$dim2_matrix" ]]; then
            echo "  Error: Matrices not generated for $base_name (order 3)"
            continue
        fi
        # Run TTM chain test for 3D
        echo "  Running TTM chain test for $base_name (order 3)..."
        ttm_output=$("$TTM_TEST" --dev 48 "$tensor_file" "$dim2_matrix" "$dim1_matrix" -l 2 2>&1)
    elif [[ "$tensor_order" -eq 4 ]]; then
        if [[ ! -f "$dim1_matrix" || ! -f "$dim2_matrix" || ! -f "$dim3_matrix" ]]; then
            echo "  Error: Matrices not generated for $base_name (order 4)"
            continue
        fi
        # Run TTM chain test for 4D
        echo "  Running TTM chain test for $base_name (order 4)..."
        ttm_output=$("$TTM_TEST" --dev 48 "$tensor_file" "$dim3_matrix" "$dim2_matrix" "$dim1_matrix" -l 3 2>&1)
    else
        echo "  Error: Unsupported tensor order $tensor_order for $base_name"
        continue
    fi
    ttm_exit_code=$?

    # Log the output for debugging
    echo "=== $base_name ===" >> "$LOG_FILE"
    echo "$ttm_output" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    if [[ $ttm_exit_code -eq 0 ]]; then
        # Extract TTM kernel times
        total_time=$(extract_ttm_times "$ttm_output")
        total_time=$(echo "$total_time / 10" | bc -l) #the kernel is runnning 10 times
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
