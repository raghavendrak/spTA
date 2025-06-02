#!/bin/bash

# Script to run tensor contractions on all CSF files
# Usage:
#
# 1. Run contractions on all CSF files in current directory:
#    ./run_contractions.sh
#
# 2. Run contractions on all CSF files in specific directory:
#    ./run_contractions.sh /path/to/csf/files
#
# 3. Run contractions on a single CSF file:
#    ./run_contractions.sh /path/to/specific/file.csf
#
# 4. Run with custom rank values:
#    RANK_1=50 RANK_2=40 ./run_contractions.sh -r1 30 -r2 60
#
# Note: Script requires NVIDIA GPU and CUDA toolkit to be installed
# as it compiles and runs the CUDA-based ttmc_gpu program

rm -f ttmc_gpu
# Ensure ttmc_gpu is compiled
if [ ! -f "./ttmc_gpu" ]; then
    echo "Building ttmc_gpu..."
    nvcc -O3 -arch=sm_70 ttmc_gpu.cu -o ttmc_gpu
    if [ $? -ne 0 ]; then
        echo "Error compiling ttmc_gpu"
        exit 1
    fi
fi

# Default rank values for factor matrices
RANK_1=30
RANK_2=30

# Function to run contractions on a single CSF file
run_contractions() {
    local csf_file="$1"
    local base_name=$(basename "$csf_file" .csf)
    # local log_file="${base_name}_contractions.log"
    local log_file="TTMC.log"
    
    echo "Running contractions on $csf_file..."
    
    # Open log file
    echo "Contraction results for $csf_file" >> "$log_file"
    echo "Timestamp: $(date)" >> "$log_file"
    echo "---------------------------------" >> "$log_file"
    
    # Run all three contraction types
    for ncm in 0 1 2; do
        echo "  Running contraction type $ncm..."
        echo "Contraction type $ncm:" >> "$log_file"
        ./ttmc_gpu "$csf_file" "$RANK_1" "$RANK_2" "$ncm" >> "$log_file" 2>&1
        echo "---------------------------------" >> "$log_file"
    done
    
    echo "  Results saved to $log_file"
}

# Process all CSF files
process_all_csf_files() {
    local dir="$1"
    
    # If no directory is provided, use current directory
    if [ -z "$dir" ]; then
        dir="."
    fi
    
    # Find all .csf files in the directory
    echo "Processing all .csf files in $dir..."
    find "$dir" -maxdepth 1 -name "*.csf" | while read -r file; do
        run_contractions "$file"
    done
}

# Main execution
echo "Tensor Contraction Runner"
echo "========================"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -r1|--rank1)
            RANK_1="$2"
            shift 2
            ;;
        -r2|--rank2)
            RANK_2="$2"
            shift 2
            ;;
        *)
            # Assume it's a file
            CSF_FILE="$1"
            shift
            ;;
    esac
done

if [ -n "$CSF_FILE" ]; then
    # Run contractions on specific file
    run_contractions "$CSF_FILE"
else
    # Run contractions on all .csf files
    process_all_csf_files "."
fi

echo "All contractions completed" 