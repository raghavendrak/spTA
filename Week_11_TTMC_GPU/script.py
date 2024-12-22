import subprocess
import csv
import re

# Configuration
compile_command = (
    "nvcc -o ttmc_gpu ttmc_gpu.cu genten.c COO_to_CSF.cpp -Xcompiler -fopenmp --extended-lambda"
)
run_command_template = (
    "./ttmc_gpu 3 1000 1000 1000 30 30 {contraction} -d {density} -f 0.1 -c 0.5 -v 0.5"
)
densities = [0.1, 0.01, 0.001]
contractions = [0, 1, 2]
iterations = 5
output_file = "results.csv"

# Compile the code
print("Compiling the code...")
subprocess.run(compile_command, shell=True, check=True)
print("Compilation complete.")

# Helper function to extract execution times
def extract_times(output):
    cpu_5_loop = re.search(r"Time taken by CPU Method - 1 \[5-for loop\].*?: ([\d.]+) seconds", output)
    cpu_4_loop = re.search(r"Time taken by CPU Method - 2 \[4-for loop\].*?: ([\d.]+) seconds", output)
    gpu_5_loop = re.search(r"Time taken by GPU Method - 1 \[5-for loop\].*?: ([\d.]+) seconds", output)
    gpu_4_loop = re.search(r"Time taken by GPU Method - 2 \[4-for loop\].*?: ([\d.]+) seconds", output)

    return (
        float(cpu_5_loop.group(1)) if cpu_5_loop else None,
        float(cpu_4_loop.group(1)) if cpu_4_loop else None,
        float(gpu_5_loop.group(1)) if gpu_5_loop else None,
        float(gpu_4_loop.group(1)) if gpu_4_loop else None,
    )

# Initialize CSV file
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Density",
        "Contraction",
        "CPU Execution Time [5-for loop]",
        "CPU Execution Time [4-for loop]",
        "GPU Execution Time [5-for loop]",
        "GPU Execution Time [4-for loop]",
    ])

    # Run experiments
    for contraction in contractions:
        for density in densities:
            print(f"Running for density={density}, contraction={contraction}...")

            cpu_5_loop_times = []
            cpu_4_loop_times = []
            gpu_5_loop_times = []
            gpu_4_loop_times = []

            for i in range(iterations):
                print(f"Iteration {i + 1}...")
                run_command = run_command_template.format(contraction=contraction, density=density)
                result = subprocess.run(run_command, shell=True, capture_output=True, text=True)

                if result.returncode != 0:
                    print("Error during execution:", result.stderr)
                    continue

                cpu_5, cpu_4, gpu_5, gpu_4 = extract_times(result.stdout)

                if cpu_5: cpu_5_loop_times.append(cpu_5)
                if cpu_4: cpu_4_loop_times.append(cpu_4)
                if gpu_5: gpu_5_loop_times.append(gpu_5)
                if gpu_4: gpu_4_loop_times.append(gpu_4)

                # Print execution times for the current iteration
                print(f"Iteration {i + 1} Results (Density={density}, Contraction={contraction}):")
                print(f"  CPU Method - 1 [5-for loop]: {cpu_5} seconds")
                print(f"  CPU Method - 2 [4-for loop]: {cpu_4} seconds")
                print(f"  GPU Method - 1 [5-for loop]: {gpu_5} seconds")
                print(f"  GPU Method - 2 [4-for loop]: {gpu_4} seconds")

            # Calculate averages
            avg_cpu_5 = round(sum(cpu_5_loop_times) / len(cpu_5_loop_times), 3) if cpu_5_loop_times else None
            avg_cpu_4 = round(sum(cpu_4_loop_times) / len(cpu_4_loop_times), 3) if cpu_4_loop_times else None
            avg_gpu_5 = round(sum(gpu_5_loop_times) / len(gpu_5_loop_times), 3) if gpu_5_loop_times else None
            avg_gpu_4 = round(sum(gpu_4_loop_times) / len(gpu_4_loop_times), 3) if gpu_4_loop_times else None

            # Write to CSV
            writer.writerow([
                density,
                contraction,
                avg_cpu_5,
                avg_cpu_4,
                avg_gpu_5,
                avg_gpu_4,
            ])

print(f"Results saved to {output_file}.")
