#!/bin/bash

# Compilation step
echo "Compiling ttmc_cpu.cpp..."
g++ -O2 -Wall -fopenmp ttmc_cpu.cpp genten.c COO_to_CSF.cpp -o ttmc_cpu
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

# Parameters
order=3
dim_0=3000
dim_1=3000
dim_2=3000
f1=30
f2=30
densities=(0.1 0.01 0.001)
contractions=(0 1 2)
num_executions=10

# Function to run and time the program
run_program() {
    density=$1
    contraction=$2
    total_seconds_1=0
    total_seconds_2=0

    for ((i=1; i<=num_executions; i++))
    do
        echo "Run $i: ./ttmc_cpu $order $dim_0 $dim_1 $dim_2 $f1 $f2 $contraction -d $density"
        
        # Capture the output from the program
        output=$(./ttmc_cpu $order $dim_0 $dim_1 $dim_2 $f1 $f2 $contraction -d $density)
        
        # Extract seconds_1 and seconds_2 values from output
        seconds_1=$(echo "$output" | grep "Time taken by contraction 1" | awk '{print $(NF-1)}')
        seconds_2=$(echo "$output" | grep "Time taken by contraction 2" | awk '{print $(NF-1)}')

        echo "Time taken by contraction 1 for Run $i : $seconds_1 sec"
        echo "Time taken by contraction 2 for Run $i : $seconds_2 sec"
        
        # Accumulate times
        total_seconds_1=$(echo "$total_seconds_1 + $seconds_1" | bc)
        total_seconds_2=$(echo "$total_seconds_2 + $seconds_2" | bc)
    done

    echo "Total Time taken by contraction 1 for $num_executions runs : $total_seconds_1 sec"
    echo "Total Time taken by contraction 2 for $num_executions runs : $total_seconds_2 sec"

    # Calculate average runtimes
    avg_seconds_1=$(echo "scale=2; $total_seconds_1 / $num_executions" | bc)
    avg_seconds_2=$(echo "scale=2; $total_seconds_2 / $num_executions" | bc)
    echo "Density $density, Contraction $contraction: "

    echo "Average Time taken by contraction 1 for $num_executions runs : $avg_seconds_1 sec"
    echo "Average Time taken by contraction 2 for $num_executions runs : $avg_seconds_2 sec"
    
    # Save to CSV
    echo "$density, $contraction, $avg_seconds_1, $avg_seconds_2" >> results.csv
}

# Create a CSV file for results
echo "Density, Contraction, Average_seconds_1, Average_seconds_2" > results.csv

# Loop through densities and contraction types
for contraction in "${contractions[@]}"; do
    for density in "${densities[@]}"; do
        run_program $density $contraction
    done
done

echo "Results saved to results.csv"
