#!/bin/bash

# Script to compile and run B-CSF for all datasets in the tensors_dataset folder
# Usage: ./run_bcsf_all_datasets.sh [options]
# 
# Options:
#   -m <mode>           : MTTKRP mode (default: 0)
#   -R <rank>           : Rank R parameter (default: 30)
#   -S <rank>           : Rank S parameter (default: 30)
#   -3 <rank>           : Rank R3 parameter (default: 30)
#   -T <flag>           : TTMC flag (default: 1)
#   -t <type>           : Implementation type (default: 8 for B-CSF)
#   -f <threshold>      : Fiber threshold (default: 128)
#   -c <flag>           : Correctness check (default: 1)
#   -v <flag>           : Verbose mode (default: 0)
#   -o <output_dir>     : Output directory for results (default: /home/bhaskar/spTA/Week_11_TTMC_GPU/bcsf_results)
#   -h                  : Show this help message

# Default values
DEFAULT_MODE=0
DEFAULT_R=30
DEFAULT_S=30
DEFAULT_R3=30
DEFAULT_T=1
DEFAULT_TYPE=8
DEFAULT_F=128
DEFAULT_C=1
DEFAULT_V=1
DEFAULT_OUTPUT_DIR="/home/bhaskar/spTA/Week_11_TTMC_GPU/"

# Parse command line arguments
while getopts "m:R:S:3:T:t:f:c:v:o:h" opt; do
    case $opt in
        m) MODE="$OPTARG" ;;
        R) R="$OPTARG" ;;
        S) S="$OPTARG" ;;
        3) R3="$OPTARG" ;;
        T) T="$OPTARG" ;;
        t) TYPE="$OPTARG" ;;
        f) F="$OPTARG" ;;
        c) C="$OPTARG" ;;
        v) V="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        h) 
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -m <mode>           : MTTKRP mode (default: $DEFAULT_MODE)"
            echo "  -R <rank>           : Rank R parameter (default: $DEFAULT_R)"
            echo "  -S <rank>           : Rank S parameter (default: $DEFAULT_S)"
            echo "  -3 <rank>           : Rank R3 parameter (default: $DEFAULT_R3)"
            echo "  -T <flag>           : TTMC flag (default: $DEFAULT_T)"
            echo "  -t <type>           : Implementation type (default: $DEFAULT_TYPE)"
            echo "  -f <threshold>      : Fiber threshold (default: $DEFAULT_F)"
            echo "  -c <flag>           : Correctness check (default: $DEFAULT_C)"
            echo "  -v <flag>           : Verbose mode (default: $DEFAULT_V)"
            echo "  -o <output_dir>     : Output directory for results (default: $DEFAULT_OUTPUT_DIR)"
            echo "  -h                  : Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -R 50 -S 50 -3 50 -v 1"
            echo "  $0 -m 1 -R 20 -S 20 -3 20 -T 0"
            exit 0
            ;;
        \?) 
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# Set default values if not provided
MODE=${MODE:-$DEFAULT_MODE}
R=${R:-$DEFAULT_R}
S=${S:-$DEFAULT_S}
R3=${R3:-$DEFAULT_R3}
T=${T:-$DEFAULT_T}
TYPE=${TYPE:-$DEFAULT_TYPE}
F=${F:-$DEFAULT_F}
C=${C:-$DEFAULT_C}
V=${V:-$DEFAULT_V}
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Paths
TENSOR_DIR="/home/bhaskar/tensors_dataset"
# SOURCE_FILE="bcsf_3.cu"
SOURCE_FILE="bcsf_complete.cu"
EXECUTABLE="bcsf"
LOG_FILE="$OUTPUT_DIR/bcsf_all_datasets.log"
RESULTS_FILE="$OUTPUT_DIR/bcsf_results.csv"

# Clear previous results
> "$LOG_FILE"
> "$RESULTS_FILE"

# Show initial message on terminal
echo "Starting B-CSF benchmark for all datasets..."
echo "Log file: $LOG_FILE"
echo "Results CSV: $RESULTS_FILE"
echo ""

echo "=== B-CSF All Datasets Benchmark ===" >> "$LOG_FILE"
echo "Started at: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

echo "Parameters:" >> "$LOG_FILE"
echo "  Mode: $MODE" >> "$LOG_FILE"
echo "  Rank R: $R" >> "$LOG_FILE"
echo "  Rank S: $S" >> "$LOG_FILE"
echo "  Rank R3: $R3" >> "$LOG_FILE"
echo "  TTMC flag: $T" >> "$LOG_FILE"
echo "  Implementation type: $TYPE" >> "$LOG_FILE"
echo "  Fiber threshold: $F" >> "$LOG_FILE"
echo "  Correctness check: $C" >> "$LOG_FILE"
echo "  Verbose mode: $V" >> "$LOG_FILE"
echo "  Output directory: $OUTPUT_DIR" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Function to compile the program
compile_program() {
    echo "Compiling B-CSF program..." >> "$LOG_FILE"
    
    # Check if source file exists
    if [[ ! -f "$SOURCE_FILE" ]]; then
        echo "Error: Source file $SOURCE_FILE not found!" >> "$LOG_FILE"
        echo "Error: Source file $SOURCE_FILE not found!"
        exit 1
    fi
    
    # Compile with nvcc
    echo "Running: nvcc -O3 -arch=sm_70 $SOURCE_FILE -o $EXECUTABLE" >> "$LOG_FILE"
    nvcc -O3 -arch=sm_70 "$SOURCE_FILE" -o "$EXECUTABLE" 2>&1 >> "$LOG_FILE"
    
    if [[ $? -ne 0 ]]; then
        echo "Error: Compilation failed!" >> "$LOG_FILE"
        echo "Error: Compilation failed!"
        exit 1
    fi
    
    echo "Compilation successful!" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
}

# Function to extract execution time from output
extract_time() {
    local output="$1"
    local time_pattern="B-CSF-GPU-mode [0-9]+ :([0-9]+\.[0-9]+) ms"
    
    if [[ "$output" =~ $time_pattern ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "ERROR"
    fi
}

# Function to extract correctness from output
extract_correctness() {
    local output="$1"
    
    if [[ "$output" =~ "Correctness pass!" ]]; then
        echo "PASS"
    elif [[ "$output" =~ "Correctness fail!" ]] || [[ "$output" =~ "ERROR" ]]; then
        echo "FAIL"
    else
        echo "UNKNOWN"
    fi
}



# Function to run B-CSF on a single dataset
run_bcsf_dataset() {
    local dataset_file="$1"
    local base_name=$(basename "$dataset_file" .tns)
    
    echo "Processing dataset: $base_name" >> "$LOG_FILE"
    echo "  File: $dataset_file" >> "$LOG_FILE"
    
    # Build command
    local cmd="./$EXECUTABLE -i \"$dataset_file\" -m $MODE -R $R -S $S -3 $R3 -T $T -t $TYPE -f $F -c $C"
    if [[ $V -eq 1 ]]; then
        cmd="$cmd -v 1"
    fi
    
    echo "  Command: $cmd" >> "$LOG_FILE"
    
    # Run the command
    echo "  Running B-CSF..." >> "$LOG_FILE"
    local output=$(eval "$cmd" 2>&1)
    local exit_code=$?
    
    # Log the complete output to log file
    echo "  Complete output:" >> "$LOG_FILE"
    echo "$output" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    # Extract results
    local execution_time=$(extract_time "$output")
    local correctness=$(extract_correctness "$output")
    
    # Log results summary
    echo "  Exit code: $exit_code" >> "$LOG_FILE"
    if [[ "$execution_time" == "ERROR" ]]; then
        echo "  Result: ERROR" >> "$LOG_FILE"
        # Add to results CSV with error message
        echo "$base_name,ERROR,FAIL" >> "$RESULTS_FILE"
    else
        echo "  Execution time: ${execution_time}ms" >> "$LOG_FILE"
        echo "  Correctness: $correctness" >> "$LOG_FILE"
        # Add to results CSV with time and correctness
        echo "$base_name,$execution_time,$correctness" >> "$RESULTS_FILE"
    fi
    
    echo "  Completed $base_name" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    return $exit_code
}



# Main execution
main() {
    # Compile the program
    compile_program
    
    # Check if tensor directory exists
    if [[ ! -d "$TENSOR_DIR" ]]; then
        echo "Error: Tensor directory $TENSOR_DIR not found!" >> "$LOG_FILE"
        echo "Error: Tensor directory $TENSOR_DIR not found!"
        exit 1
    fi
    
    # Create CSV header
    echo "Dataset,Execution_Time(ms),Correctness" > "$RESULTS_FILE"
    
    # Find all .tns files
    local tensor_files=($(find "$TENSOR_DIR" -name "*.tns" -type f))
    local total_datasets=${#tensor_files[@]}
    local processed=0
    local successful=0
    local failed=0
    
    echo "Found $total_datasets tensor files" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    # Process each dataset
    for tensor_file in "${tensor_files[@]}"; do
        ((processed++))
        echo "Progress: $processed/$total_datasets" >> "$LOG_FILE"
        # Check if the first line of the tensor file is 3
        first_line=$(head -n 1 "$tensor_file")
        if [[ "$first_line" != "3" ]]; then
            echo "Skipping $(basename "$tensor_file"): First line is '$first_line' (expected '3')" >> "$LOG_FILE"
            continue
        fi
        
        if run_bcsf_dataset "$tensor_file"; then
            ((successful++))
        else
            ((failed++))
        fi
        
        echo "Progress: $processed/$total_datasets completed" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
    done
    
    # Final summary
    echo "=== Benchmark Completed ===" >> "$LOG_FILE"
    echo "Total datasets: $total_datasets" >> "$LOG_FILE"
    echo "Successful: $successful" >> "$LOG_FILE"
    echo "Failed: $failed" >> "$LOG_FILE"
    echo "Results saved to: $OUTPUT_DIR" >> "$LOG_FILE"
    echo "Completed at: $(date)" >> "$LOG_FILE"
    
    # Display results table
    echo "" >> "$LOG_FILE"
    echo "=== Results Table ===" >> "$LOG_FILE"
    if [[ -f "$RESULTS_FILE" ]]; then
        column -t -s',' "$RESULTS_FILE" >> "$LOG_FILE"
    fi
    
    # Show final summary on terminal
    echo "Benchmark completed!"
    echo "Total datasets: $total_datasets"
    echo "Successful: $successful"
    echo "Failed: $failed"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo "CSV file: $RESULTS_FILE"
}

# Run main function
main "$@"
