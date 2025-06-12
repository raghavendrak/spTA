#!/bin/bash

# Script to run tensor contractions on all CSF files with multiple methods
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
# 5. Run only a specific contraction type (0, 1, or 2):
#    ./run_contractions.sh --ncm 1
#
# 6. Run with result verification:
#    ./run_contractions.sh --verify
#
# 7. Run with all options:
#    ./run_contractions.sh /path/to/file.csf  --verbose -r1 30 -r2 60 --ncm 1 --verify
#
# Note: Script requires NVIDIA GPU and CUDA toolkit to be installed
# as it compiles and runs the CUDA-based implementations

# Clean up any previous executables
rm -f ttmc_v*

# Find all method implementation files
echo "Finding contraction methods..."
method_files=(v*.cu)
method_numbers=()
skip_methods=(1  6)  # Skip method 1 (v1_cpu_5loop.cu)

for file in "${method_files[@]}"; do
    if [[ $file =~ v([0-9]+)_ ]]; then
        method_num=${BASH_REMATCH[1]}
        # Skip methods listed in skip_methods
        skip=false
        for skip_method in "${skip_methods[@]}"; do
            if [[ $method_num -eq $skip_method ]]; then
                skip=true
                echo "Skipping method $method_num: $file"
                break
            fi
        done
        if [[ $skip == false ]]; then
            method_numbers+=($method_num)
        fi
    fi
done

# Sort method numbers
IFS=$'\n' method_numbers=($(sort -n <<<"${method_numbers[*]}"))
unset IFS

echo "Found methods: ${method_numbers[@]}"

# Compile each implementation file
echo "Building contraction executables..."
for method in ${method_numbers[@]}; do
    input_file="v${method}_"*.cu
    output="ttmc_v${method}"
    echo "  Compiling $input_file -> $output"
    nvcc -O3 -arch=sm_70 $input_file -o $output
    if [ $? -ne 0 ]; then
        echo "Error compiling $output"
    fi
done

# Default rank values for factor matrices
RANK_1=30
RANK_2=30
NCM="" # Empty means run all contraction types
VERBOSE=false
CSF_FILE=""
VERIFY=false  # Default: don't verify results
RUNS=1        # Default: run each method once

# Parse command-line arguments
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
        -n|--ncm)
            NCM="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --verify)
            VERIFY=true
            shift
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        *.csf)
            # If argument ends with .csf, it's a CSF file
            CSF_FILE="$1"
            shift
            ;;
        *)
            # Check if it might be a directory or a CSF file
            if [ -d "$1" ]; then
                CSF_DIR="$1"
                shift
            elif [ -f "$1" ] && [[ "$1" == *.csf ]]; then
                CSF_FILE="$1"
                shift
            else
                echo "Unknown option: $1"
                shift
            fi
            ;;
    esac
done

echo "Using factor ranks: R1=$RANK_1, R2=$RANK_2"
if [ "$VERBOSE" = true ]; then
    echo "Verbose mode enabled"
fi
if [ "$VERIFY" = true ]; then
    echo "Result verification enabled (first run only)"
fi
if [ $RUNS -gt 1 ]; then
    echo "Running each method $RUNS times"
fi

# Initialize logs for all contraction types if NCM is empty
initialize_logs() {
    if [ -z "$NCM" ]; then
        for ncm in 0 1 2; do
            local log_file="TTMC_ncm_${ncm}.log"
            echo "Tensor contraction results" > "$log_file"
            echo "Timestamp: $(date)" >> "$log_file"
            echo "Using factor matrix ranks: R1=$RANK_1, R2=$RANK_2" >> "$log_file"
            echo "Number of runs per method: $RUNS" >> "$log_file"
            echo "---------------------------------" >> "$log_file"
        done
    else
        local log_file="TTMC_ncm_${NCM}.log"
        echo "Tensor contraction results" > "$log_file"
        echo "Timestamp: $(date)" >> "$log_file"
        echo "Using factor matrix ranks: R1=$RANK_1, R2=$RANK_2" >> "$log_file"
        echo "Number of runs per method: $RUNS" >> "$log_file"
        echo "---------------------------------" >> "$log_file"
    fi
}

# Function to run contractions on a single CSF file
run_contractions() {
    local csf_file="$1"
    local base_name=$(basename "$csf_file")
    
    echo "Running contractions on $csf_file..."
    
    # Run all contraction methods
    for method in ${method_numbers[@]}; do
        executable="ttmc_v${method}"
        
        if [ -f "$executable" ]; then
            echo "  Running method v${method}..."
            
            # Build the command with appropriate options
            local cmd_options=()
            
            # Add rank options
            cmd_options+=("-r1" "$RANK_1" "-r2" "$RANK_2")
            
            # If NCM is specified, run only that contraction type
            # Otherwise, run all three contraction types (0, 1, 2)
            if [ -n "$NCM" ]; then
                # Run only the specified contraction type
                local log_file="TTMC_ncm_${NCM}.log"
                echo "Running contraction on $csf_file..." >> "$log_file"
                
                if [ "$NCM" == 0 ]; then
                    echo "Your Contraction Choice : ijk,jr,ks→irs" >> "$log_file"
                elif [ "$NCM" == 1 ]; then
                    echo "Your Contraction Choice : ijk,ir,ks→rjs" >> "$log_file"
                elif [ "$NCM" == 2 ]; then
                    echo "Your Contraction Choice : ijk,ir,js→rsk" >> "$log_file"
                fi
                
                # Add NCM option
                cmd_options+=("-n" "$NCM")
                
                # Add verbose flag if needed
                if [ "$VERBOSE" = true ]; then
                    cmd_options+=("-v")
                fi
                
                # Run the executable multiple times if requested
                for run in $(seq 1 $RUNS); do
                    echo "Run $run/$RUNS of method v${method}..." >> "$log_file"
                    
                    # Only verify on first run
                    local run_options=("${cmd_options[@]}")
                    if [ "$VERIFY" = true ] && [ $run -eq 1 ]; then
                        run_options+=("--verify")
                    fi
                    
                    # Run the executable with all options
                    ./$executable "$csf_file" "${run_options[@]}" >> "$log_file" 2>&1
                done
                
                echo "---------------------------------" >> "$log_file"
            else
                # Run all three contraction types
                for ncm in 0 1 2; do
                    local log_file="TTMC_ncm_${ncm}.log"
                    echo "Running contraction type $ncm on $csf_file..." >> "$log_file"
                    
                    # Add NCM option
                    local ncm_cmd_options=("${cmd_options[@]}" "-n" "$ncm")
                    
                    # Add verbose flag if needed
                    if [ "$VERBOSE" = true ]; then
                        ncm_cmd_options+=("-v")
                    fi
                    
                    # Run the executable multiple times if requested
                    for run in $(seq 1 $RUNS); do
                        echo "Run $run/$RUNS of method v${method}..." >> "$log_file"
                        
                        # Only verify on first run
                        local run_options=("${ncm_cmd_options[@]}")
                        if [ "$VERIFY" = true ] && [ $run -eq 1 ]; then
                            run_options+=("--verify")
                        fi
                        
                        # Run the executable with all options
                        ./$executable "$csf_file" "${run_options[@]}" >> "$log_file" 2>&1
                    done
                    
                    echo "---------------------------------" >> "$log_file"
                done
            fi
        else
            echo "  Executable $executable not found, skipping..."
        fi
    done
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

# Generate performance plots
generate_plots() {
    echo "Generating performance plots..."
    if [ -n "$NCM" ]; then
        python parse_logs.py "TTMC_ncm_${NCM}.log" -o "speedup_ncm_${NCM}.png" -s "v2"
    else
        for ncm in 0 1 2; do
            python parse_logs.py "TTMC_ncm_${ncm}.log" -o "speedup_ncm_${ncm}.png" -s "v2"
        done
    fi
}

# Main execution
echo "Tensor Contraction Runner"
echo "========================"

if [ -n "$NCM" ]; then
    echo "Running only contraction type $NCM"
else
    echo "Running all contraction types (0, 1, 2)"
fi


# Initialize log files
initialize_logs

if [ -n "$CSF_FILE" ]; then
    # Run contractions on specific file
    run_contractions "$CSF_FILE"
elif [ -n "$CSF_DIR" ]; then
    # Run contractions on all .csf files in specified directory
    process_all_csf_files "$CSF_DIR"
else
    # Run contractions on all .csf files in current directory
    process_all_csf_files "."
fi

# Generate plots after all contractions complete
generate_plots

echo "All contractions completed" 