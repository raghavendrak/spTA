#!/bin/bash

# Set the input parameters
order=3
dim_0=2000
dim_1=2000
dim_2=2000
other_args="-f 0.1 -c 0.5 -v 0.5"

# Compile the program
nvcc -o cootocsfgpu COO_to_CSF_GPU.cu genten.o -Xcompiler -fopenmp --extended-lambda

# Output file to save results
output_file="results_gpu.csv"

# Write CSV header to the file
echo "Density,CPU_AverageExecutionTime(Seconds), GPU_AverageExecutionTime(Seconds)" > $output_file

# Loop through different densities
for density in 0.01 0.001 0.0001
do
    echo "Running for density = $density"

    # Add the density value to the arguments
    density_arg="-d $density"

    # Initialize sum of execution times for averaging
    cpu_total_time=0
    gpu_total_time=0


    # Run the program 10 times and accumulate execution times
    for i in {1..10}
    do
        echo "  Running iteration $i..."

        # Run the program and capture its output
        output=$(./cootocsfgpu $order $dim_0 $dim_1 $dim_2 $density_arg $other_args)

        # Extract the execution time from the output using grep and awk    
        cpu_exec_time=$(echo "$output" | grep "COO to CSF CPU execution time:" | awk '{print $7}') 
        gpu_exec_time=$(echo "$output" | grep "COO to CSF GPU execution time:" | awk '{print $7}')

        # Output execution time to the console
        echo " CPU Execution time for iteration $i: $cpu_exec_time seconds"
        echo " GPU Execution time for iteration $i: $gpu_exec_time seconds"

        # Add the execution time to total_time
        cpu_total_time=$(echo "$cpu_total_time + $cpu_exec_time" | bc)
        gpu_total_time=$(echo "$gpu_total_time + $gpu_exec_time" | bc)
    done

    # Calculate the average execution time for this density
    cpu_avg_time=$(echo "$cpu_total_time / 10" | bc -l)
    gpu_avg_time=$(echo "$gpu_total_time / 10" | bc -l)

    # Output average execution time for the density to the console
    echo " CPU Average execution time for density $density: $cpu_avg_time_6dp seconds"
    echo " GPU Average execution time for density $density: $gpu_avg_time_6dp seconds"

    # Append the density and average execution time to the CSV file
    echo "$density,$cpu_avg_time,$gpu_avg_time" >> $output_file
done

echo "Results saved to $output_file"
