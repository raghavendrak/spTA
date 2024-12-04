#!/bin/bash

# Set the input parameters
order=3
dim_0=2000
dim_1=2000
dim_2=2000
other_args="-f 0.1 -c 0.5 -v 0.5 -o ../sample_data/generated_100_3D.tns"

# Compile the program
g++ -O2 -Wall -fopenmp COO_to_CSF.cpp genten.c -o cootocsf

# Output file to save results
output_file="results_cpu.csv"

# Write CSV header to the file
echo "Density,AverageExecutionTime(Seconds)" > $output_file

# Loop through different densities
for density in 0.01 0.001 0.0001
do
    echo "Running for density = $density"

    # Add the density value to the arguments
    density_arg="-d $density"

    # Initialize sum of execution times for averaging
    total_time=0

    # Run the program 10 times and accumulate execution times
    for i in {1..10}
    do
        echo "  Running iteration $i..."

        # Run the program and capture its output
        output=$(./cootocsf $order $dim_0 $dim_1 $dim_2 $density_arg $other_args)

        # Extract the execution time from the output using grep and awk
        exec_time=$(echo "$output" | grep "COO to CSF CPU execution time:" | awk '{print $7}')

        # Output execution time to the console
        echo "  Execution time for iteration $i: $exec_time seconds"

        # Add the execution time to total_time
        total_time=$(echo "$total_time + $exec_time" | bc)
    done

    # Calculate the average execution time for this density
    avg_time=$(echo "$total_time / 10" | bc -l)

    # Output average execution time for the density to the console
    echo "  Average execution time for density $density: $avg_time_6dp seconds"

    # Append the density and average execution time to the CSV file
    echo "$density,$avg_time" >> $output_file
done

echo "Results saved to $output_file"
