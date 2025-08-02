#!/bin/bash

# Script to process .tns files by:
# 1. Running preprocessor.py to add tensor order and dimensions
# 2. Converting to CSF format using COO_to_CSF_file

# Configuration - Set the path to your tensors dataset directory
TENSORS_PATH="../../tensors_dataset/4d_tensors"  # Change this to your tensors directory path

rm COO_to_CSF_file
# Ensure COO_to_CSF_file is compiled
if [ ! -f "./COO_to_CSF_file" ]; then
    echo "Building COO_to_CSF_file..."
    g++ -std=c++11 -O3 -fopenmp COO_to_CSF_file.cpp -o COO_to_CSF_file
fi

# Function to process a single tensor file
process_tensor() {
    local tns_file="$1"
    
    # Skip if not a .tns file with exact extension
    # if [[ "$tns_file" != *sib_3d.tns ]]; then
    if [[ "$tns_file" != *.tns ]]; then
        echo "Skipping $tns_file (not a .tns file)"
        return
    fi
    
    local base_name=$(basename "$tns_file" .tns)
    local csf_file="${base_name}.csf"
    
    echo "Processing $tns_file..."
    
    # Run preprocessor.py on the .tns file
    echo "  Running preprocessor..."
    #comment below line if your COO file already have tensor order and tensor dimensions written
    # python preprocesor.py "$tns_file"
    if [ $? -ne 0 ]; then
        echo "  Error preprocessing $tns_file"
        return
    fi
    
    # Run COO_to_CSF_file to convert to CSF format
    echo "  Converting to CSF format..."
    ./COO_to_CSF_file "$tns_file" "$csf_file"
    if [ $? -ne 0 ]; then
        echo "  Error converting $tns_file to CSF"
        return
    fi
    
    echo "  Successfully created $csf_file"
}

# Process each tensor file
process_all_tensors() {
    local tensor_dir="$1"
    
    # If no directory is provided, use the configured TENSORS_PATH
    if [ -z "$tensor_dir" ]; then
        tensor_dir="$TENSORS_PATH"
    fi
    
    # Check if tensor directory exists
    if [ ! -d "$tensor_dir" ]; then
        echo "Error: Tensor directory '$tensor_dir' does not exist!"
        exit 1
    fi
    
    # Find all .tns files (exact extension match) in the directory
    echo "Processing all .tns files in $tensor_dir..."
    find "$tensor_dir" -maxdepth 1 -name "*.tns" | while read -r file; do
        process_tensor "$file"
    done
}

# Main execution
echo "Tensor Processing Script"
echo "======================="

if [ -n "$1" ]; then
    if [ -d "$1" ]; then
        # Process all files in specified directory
        process_all_tensors "$1"
    else
        # Process specific file if provided
        process_tensor "$1"
    fi
else
    # Process all tensors in default directory
    process_all_tensors
fi

echo "All processing completed" 