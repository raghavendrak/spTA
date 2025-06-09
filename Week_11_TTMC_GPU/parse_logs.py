#!/usr/bin/env python3
# Usage:
#
# 1. Parse log file and generate speedup plot with default output filename:
#    python parse_logs.py TTMC_ncm_2.log
#
# 2. Parse log file and specify custom output filename:
#    python parse_logs.py TTMC_ncm_2.log -o custom_speedup.png
#
# 3. Parse log file and save plot to specific directory:
#    python parse_logs.py TTMC_ncm_2.log -o /path/to/dir/speedup.png
#
# 4. Parse log file and specify baseline method:
#    python parse_logs.py TTMC_ncm_2.log -b gpu_4loop_streams
#
# 5. Parse log file and specify y-axis maximum value:
#    python parse_logs.py TTMC_ncm_2.log -y 15.0
#
# 6. Parse log file and specify baseline method and y-axis maximum value:
#    python parse_logs.py TTMC_ncm_2.log -b gpu_4loop_streams -y 15.0
#
# 7. Parse log file and skip specific methods:
#    python parse_logs.py TTMC_ncm_2.log -s cpu_4loop gpu_5loop
#
# The script parses TTMC log files containing timing data for tensor contractions
# and generates a bar plot comparing speedups of different GPU implementations
# relative to the CPU baseline implementation.
#
# Required format of log file:
# - Contains sections starting with "Contraction results for <dataset>"
# - Each section has timing data for CPU and GPU implementations
# - Timing data in format "Time taken by <method> : <time> milliseconds"

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import MaxNLocator
from collections import defaultdict

def parse_log_file(log_file_path):
    """Parse the TTMC log file and extract timing data for each dataset"""
    
    results = []
    contraction_choice = None
    
    with open(log_file_path, 'r') as f:
        content = f.read()
        
    # Extract contraction choice from the beginning of the file
    contraction_match = re.search(r"Your Contraction Choice : (.*?)\n", content)
    if contraction_match:
        contraction_choice = contraction_match.group(1)
    
    # Dictionary to store timing information by dataset name
    dataset_timings = defaultdict(lambda: defaultdict(list))
    
    # Find all dataset entries
    dataset_entries = re.finditer(r"Running contraction on (.*?)\.\.\.?\n", content)
    
    # Process each dataset entry
    for dataset_entry in dataset_entries:
        dataset_name = dataset_entry.group(1).strip()
        # Extract base dataset name without path and extension
        base_name = os.path.basename(dataset_name)
        base_name = os.path.splitext(base_name)[0]
        
        # Find the starting point of this dataset entry
        start_pos = dataset_entry.start()
        
        # Find the end point (next dataset entry or end of content)
        next_dataset = re.search(r"Running contraction on (.*?)\.\.\.?\n", content[start_pos + 1:])
        if next_dataset:
            end_pos = start_pos + 1 + next_dataset.start()
        else:
            end_pos = len(content)
        
        # Extract the section for this dataset
        dataset_section = content[start_pos:end_pos]
        
        # Find all method runs in this section
        
        # CPU 4-loop (reference)
        cpu_times = re.findall(r"Method: CPU 4-loop \(reference\), Time: (\d+(?:\.\d+)?) ms", dataset_section)
        for time in cpu_times:
            dataset_timings[base_name]['cpu_4loop'].append(float(time))
        
        # GPU 5-loop
        gpu5_times = re.findall(r"Method: GPU 5-loop, Time: (\d+(?:\.\d+)?) ms", dataset_section)
        for time in gpu5_times:
            dataset_timings[base_name]['gpu_5loop'].append(float(time))
        
        # GPU 4-loop
        gpu4_times = re.findall(r"Method: GPU 4-loop, Time: (\d+(?:\.\d+)?) ms", dataset_section)
        for time in gpu4_times:
            dataset_timings[base_name]['gpu_4loop'].append(float(time))
        
        # GPU 4-loop streams
        gpu4s_times = re.findall(r"Method: GPU 4-loop streams, Time: (\d+(?:\.\d+)?) ms", dataset_section)
        for time in gpu4s_times:
            dataset_timings[base_name]['gpu_4loop_streams'].append(float(time))
    
    # Calculate the maximum number of runs found for any method
    max_runs = 1
    for base_name, timings in dataset_timings.items():
        for method, times in timings.items():
            if len(times) > max_runs:
                max_runs = len(times)
    
    print(f"Detected maximum of {max_runs} runs per method")
    
    # Calculate average times and convert dictionary to list
    for base_name, timings in dataset_timings.items():
        # Skip empty datasets
        if all(len(times) == 0 for times in timings.values()):
            print(f"Skipping dataset {base_name} because it has no timing data")
            continue
        
        result = {'name': base_name}
        
        # Calculate average for each method
        for method, times in timings.items():
            if times:
                # Store raw timing values
                result[f'{method}_times'] = times  # Store all raw times
                
                # Calculate average if we have times
                result[method] = np.mean(times)  # Average time
                result[f'{method}_std'] = np.std(times, ddof=1) if len(times) > 1 else 0  # Standard deviation
                result[f'{method}_runs'] = len(times)  # Number of runs
                
                # Calculate coefficient of variation (CV) for methods with multiple runs
                if len(times) > 1:
                    cv = (result[f'{method}_std'] / result[method]) * 100  # CV as percentage
                    result[f'{method}_cv'] = cv
                    
                    # Log detailed information about multiple runs
                    print(f"{base_name} - {method}: {len(times)} runs, avg={result[method]:.2f}ms, min={min(times):.2f}ms, max={max(times):.2f}ms, std={result[f'{method}_std']:.2f}, cv={cv:.1f}%")
            else:
                # No times recorded for this method
                result[method] = None
                result[f'{method}_runs'] = 0
                result[f'{method}_std'] = 0
                result[f'{method}_times'] = []
                
        results.append(result)
        
    return results, contraction_choice, max_runs

def calculate_speedups(results, baseline_method='gpu_4loop'):
    """Calculate speedup for each method with specified baseline, including error propagation"""
    
    for dataset in results:
        # Skip datasets with no baseline timing or handle them specially
        if dataset[baseline_method] is None:
            # Initialize all speedups to None since we can't calculate them without the baseline
            dataset['cpu_4loop_speedup'] = None
            dataset['gpu_5loop_speedup'] = None
            dataset['gpu_4loop_speedup'] = None
            dataset['gpu_4loop_streams_speedup'] = None
            continue
            
        baseline = dataset[baseline_method]
        baseline_std = dataset.get(f'{baseline_method}_std', 0)
        
        # Calculate speedups (baseline / method_time)
        if dataset['cpu_4loop'] is not None:
            dataset['cpu_4loop_speedup'] = baseline / dataset['cpu_4loop']
            
            # Error propagation for division: relative errors add in quadrature
            if dataset['cpu_4loop_runs'] > 1 and dataset[f'{baseline_method}_runs'] > 1:
                rel_err_baseline = baseline_std / baseline if baseline > 0 else 0
                rel_err_method = dataset['cpu_4loop_std'] / dataset['cpu_4loop'] if dataset['cpu_4loop'] > 0 else 0
                rel_err_speedup = np.sqrt(rel_err_baseline**2 + rel_err_method**2)
                dataset['cpu_4loop_speedup_std'] = dataset['cpu_4loop_speedup'] * rel_err_speedup
            else:
                dataset['cpu_4loop_speedup_std'] = 0
        else:
            dataset['cpu_4loop_speedup'] = None
            dataset['cpu_4loop_speedup_std'] = 0
            
        if dataset['gpu_5loop'] is not None:
            dataset['gpu_5loop_speedup'] = baseline / dataset['gpu_5loop']
            
            # Error propagation
            if dataset['gpu_5loop_runs'] > 1 and dataset[f'{baseline_method}_runs'] > 1:
                rel_err_baseline = baseline_std / baseline if baseline > 0 else 0
                rel_err_method = dataset['gpu_5loop_std'] / dataset['gpu_5loop'] if dataset['gpu_5loop'] > 0 else 0
                rel_err_speedup = np.sqrt(rel_err_baseline**2 + rel_err_method**2)
                dataset['gpu_5loop_speedup_std'] = dataset['gpu_5loop_speedup'] * rel_err_speedup
            else:
                dataset['gpu_5loop_speedup_std'] = 0
        else:
            dataset['gpu_5loop_speedup'] = None
            dataset['gpu_5loop_speedup_std'] = 0
        
        # The baseline method's speedup is always 1.0
        dataset[f'{baseline_method}_speedup'] = 1.0
        dataset[f'{baseline_method}_speedup_std'] = 0  # Zero std dev for the baseline
            
        if baseline_method != 'gpu_4loop' and dataset['gpu_4loop'] is not None:
            dataset['gpu_4loop_speedup'] = baseline / dataset['gpu_4loop']
            
            # Error propagation
            if dataset['gpu_4loop_runs'] > 1 and dataset[f'{baseline_method}_runs'] > 1:
                rel_err_baseline = baseline_std / baseline if baseline > 0 else 0
                rel_err_method = dataset['gpu_4loop_std'] / dataset['gpu_4loop'] if dataset['gpu_4loop'] > 0 else 0
                rel_err_speedup = np.sqrt(rel_err_baseline**2 + rel_err_method**2)
                dataset['gpu_4loop_speedup_std'] = dataset['gpu_4loop_speedup'] * rel_err_speedup
            else:
                dataset['gpu_4loop_speedup_std'] = 0
        else:
            if baseline_method != 'gpu_4loop':
                dataset['gpu_4loop_speedup'] = None
                dataset['gpu_4loop_speedup_std'] = 0
            
        if baseline_method != 'gpu_4loop_streams' and dataset['gpu_4loop_streams'] is not None:
            dataset['gpu_4loop_streams_speedup'] = baseline / dataset['gpu_4loop_streams']
            
            # Error propagation
            if dataset['gpu_4loop_streams_runs'] > 1 and dataset[f'{baseline_method}_runs'] > 1:
                rel_err_baseline = baseline_std / baseline if baseline > 0 else 0
                rel_err_method = dataset['gpu_4loop_streams_std'] / dataset['gpu_4loop_streams'] if dataset['gpu_4loop_streams'] > 0 else 0
                rel_err_speedup = np.sqrt(rel_err_baseline**2 + rel_err_method**2)
                dataset['gpu_4loop_streams_speedup_std'] = dataset['gpu_4loop_streams_speedup'] * rel_err_speedup
            else:
                dataset['gpu_4loop_streams_speedup_std'] = 0
        else:
            if baseline_method != 'gpu_4loop_streams':
                dataset['gpu_4loop_streams_speedup'] = None
                dataset['gpu_4loop_streams_speedup_std'] = 0
            
    return results

def plot_speedups(results, baseline_method='gpu_4loop', contraction_choice=None, output_file=None, y_max=5, skip_methods=None, runs_per_method=1):
    """Create a bar plot showing speedups for all datasets
    
    Parameters:
    -----------
    results : list
        List of dictionaries containing timing and speedup data for each dataset
    baseline_method : str
        Method used as baseline for speedup calculations
    contraction_choice : str
        Contraction operation used in the benchmark
    output_file : str
        Path to save the output plot image
    y_max : float
        Maximum value for y-axis
    skip_methods : list
        List of method names to skip in the plot (e.g., ['cpu_4loop', 'gpu_5loop'])
    runs_per_method : int
        Number of runs performed for each method (used in plot title)
    """
    
    # Initialize skip_methods if not provided
    if skip_methods is None:
        skip_methods = []
    
    # Filter results to only include datasets with the baseline method available
    filtered_results = [r for r in results if r[baseline_method] is not None]
    
    if not filtered_results:
        print("No datasets have the baseline method available for plotting")
        return
    
    # Get the method labels
    all_method_keys = ['cpu_4loop', 'gpu_5loop', 'gpu_4loop', 'gpu_4loop_streams']
    
    # Filter out methods that should be skipped
    method_keys = [method for method in all_method_keys if method not in skip_methods and method != baseline_method]
    # Always include baseline method even if it's in skip_methods
    method_keys.append(baseline_method)
    # Remove duplicates and maintain order
    method_keys = list(dict.fromkeys(method_keys))
    
    # Extract data for plotting (only for methods we're not skipping)
    datasets = [r['name'] for r in filtered_results]
    
    # Extract speedups and standard deviations for methods we want to plot
    speedup_data = []
    std_devs = []  # Standard deviations for each method
    
    for method in method_keys:
        speedup_key = f"{method}_speedup"
        std_key = f"{method}_speedup_std"
        
        # Collect speedups and their standard deviations
        method_speedups = []
        method_stds = []
        
        for r in filtered_results:
            # Get the speedup
            speedup = r.get(speedup_key, np.nan)
            speedup = speedup if speedup is not None else np.nan
            method_speedups.append(speedup)
            
            # Get the standard deviation of the speedup (from error propagation)
            std_dev = r.get(std_key, 0)
            method_stds.append(std_dev)
        
        speedup_data.append(method_speedups)
        std_devs.append(method_stds)
    
    method_names = {
        'cpu_4loop': 'CPU 4-loop',
        'gpu_5loop': 'GPU 5-loop', 
        'gpu_4loop': 'GPU 4-loop',
        'gpu_4loop_streams': 'GPU 4-loop Streams'
    }
    
    baseline_label = method_names[baseline_method] + ' (Baseline)'
    
    # Setup figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bar positions
    x = np.arange(len(datasets))
    bar_width = 0.2
    
    # Create bars with appropriate labels
    bar_colors = ['blue', 'red', 'green', 'purple']
    
    # Calculate positions based on number of methods being plotted
    num_methods = len(method_keys)
    if num_methods <= 1:
        bar_positions = [0]  # Only baseline
    else:
        # Calculate positions to center the bars
        total_width = bar_width * (num_methods - 1)
        start_pos = -total_width / 2
        bar_positions = [start_pos + i * bar_width for i in range(num_methods)]
    
    # Create bars for each method with error bars for standard deviation
    for i, (method, data, std, pos) in enumerate(zip(method_keys, speedup_data, std_devs, bar_positions)):
        label = method_names[method]
        if method == baseline_method:
            label = baseline_label
        color = bar_colors[i % len(bar_colors)]  # Cycle through colors if needed
        
        # Plot with error bars only where std > 0
        ax.bar(x + pos, data, bar_width, label=label, color=color)
        
        # Convert standard deviations to numpy arrays for easier handling
        std_array = np.array(std)
        data_array = np.array(data)
        
        # Only add error bars where std > 0 (multiple runs with variation)
        mask = ~np.isnan(data_array) & (std_array > 0)
        if any(mask):
            # Add error bars only where we have std > 0
            ax.errorbar(
                x[mask] + pos, data_array[mask], 
                yerr=std_array[mask], 
                fmt='none', ecolor='black', capsize=3, 
                linewidth=1, alpha=0.7
            )
    
    # Add horizontal line at y=1
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    
    # Add speedup values on top of bars
    for i in range(len(datasets)):
        # Add labels for all methods
        for j, (method, data, std, pos) in enumerate(zip(method_keys, speedup_data, std_devs, bar_positions)):
            if not np.isnan(data[i]):
                height = data[i]
                # Get the run count for this dataset/method
                run_count = filtered_results[i].get(f"{method}_runs", 0)
                # Format run count and CV for display
                if run_count > 1 and std[i] > 0:
                    # Calculate CV as percentage
                    cv = (std[i] / height * 100) if height > 0 else 0
                    runs_label = ""#f"({run_count}, CV:{cv:.1f}%)" if cv > 5 else f"({run_count})"
                else:
                    runs_label = ""#f"({run_count})" if run_count > 1 else ""
                
                if height > y_max - 1:  # Adjust position for very tall bars
                    # Place label at 90% of y-axis limit
                    text_height = y_max * 0.9
                    ax.text(x[i] + pos, text_height, 
                            f"{height:.2f}x {runs_label}", ha='center', va='bottom', fontsize=8, 
                            rotation=75, color='black', weight='bold')
                else:
                    ax.text(x[i] + pos, height + 0.05*height, 
                            f"{height:.2f}x {runs_label}", ha='center', va='bottom', fontsize=8, 
                            rotation=75)
    
    # Customize plot
    ax.set_xlabel('Dataset')
    ax.set_ylabel(f'Speedup (relative to {method_names[baseline_method]})')
    
    # Set the title with contraction choice and run info if available
    title = f'TTMC Method Performance Comparison (Baseline: {method_names[baseline_method]})'
    if runs_per_method > 1:
        title += f' - Averaged from {runs_per_method} runs'
    if contraction_choice:
        title += f' - Contraction: {contraction_choice}'
    ax.set_title(title)
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Add memory error notations
    for i, dataset in enumerate(filtered_results):
        positions = bar_positions
        
        for method_idx, method in enumerate(method_keys):
            speedup_key = f"{method}_speedup"
            pos = positions[method_idx]
            if speedup_key in dataset and dataset[speedup_key] is None:
                ax.text(i + pos, 0.05, 'Mem Error', ha='center', va='bottom', rotation=90, 
                        fontsize=8, color='red')
    
    # Set a fixed y-axis limit as requested
    ax.set_ylim(0, y_max)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Parse TTMC log files and create performance plots')
    parser.add_argument('log_file', help='Path to the TTMC log file')
    parser.add_argument('-o', '--output', help='Output file path for the plot (e.g., speedup_plot.png)')
    parser.add_argument('-b', '--baseline', default='gpu_4loop', 
                        choices=['cpu_4loop', 'gpu_5loop', 'gpu_4loop', 'gpu_4loop_streams'],
                        help='Method to use as the baseline for speedup calculation')
    parser.add_argument('-y', '--y-max', type=float, default=5.0,
                        help='Maximum value for the y-axis (default: 5.0)')
    parser.add_argument('-s', '--skip', nargs='+', choices=['cpu_4loop', 'gpu_5loop', 'gpu_4loop', 'gpu_4loop_streams'],
                        help='Methods to skip in the plot (e.g., -s cpu_4loop gpu_5loop)')
    args = parser.parse_args()
    
    # Parse log file
    print(f"Parsing log file: {args.log_file}")
    results, contraction_choice, max_runs = parse_log_file(args.log_file)
    
    if not results:
        print("No valid results found in log file.")
        return
    
    if contraction_choice:
        print(f"Found contraction choice: {contraction_choice}")
    
    # Calculate speedups
    baseline_method = args.baseline
    results = calculate_speedups(results, baseline_method)
    
    # Print summary
    method_names = {
        'cpu_4loop': 'CPU 4-loop',
        'gpu_5loop': 'GPU 5-loop', 
        'gpu_4loop': 'GPU 4-loop',
        'gpu_4loop_streams': 'GPU 4-loop Streams'
    }
    
    print(f"\nUsing {method_names[baseline_method]} as baseline")
    
    if args.skip:
        print(f"Skipping methods: {', '.join(args.skip)}")
    
    print("\nPerformance Summary (Average Times):")
    print("-" * 100)
    print(f"{'Dataset':<25} {'CPU 4-loop':<20} {'GPU 5-loop':<20} {'GPU 4-loop':<20} {'GPU 4-loop Strm':<20}")
    print("-" * 100)
    
    for r in results:
        cpu = f"{r['cpu_4loop']:.2f}ms " if r['cpu_4loop'] is not None else "N/A"
        gpu5 = f"{r['gpu_5loop']:.2f}ms " if r['gpu_5loop'] is not None else "N/A"
        gpu4 = f"{r['gpu_4loop']:.2f}ms " if r['gpu_4loop'] is not None else "N/A"
        gpu4s = f"{r['gpu_4loop_streams']:.2f}ms " if r['gpu_4loop_streams'] is not None else "N/A"
        
        print(f"{r['name']:<25} {cpu:<20} {gpu5:<20} {gpu4:<20} {gpu4s:<20}")
    
    print("\nSpeedup Summary (relative to {}):".format(method_names[baseline_method]))
    print("-" * 100)
    print(f"{'Dataset':<25} {'CPU 4-loop':<15} {'GPU 5-loop':<15} {'GPU 4-loop':<15} {'GPU 4-loop Strm':<15}")
    print("-" * 100)
    
    for r in results:
        # Skip datasets with no baseline timing
        if baseline_method in r and r[baseline_method] is None:
            continue
            
        # Ensure all speedup keys exist
        speedup_keys = ['cpu_4loop_speedup', 'gpu_5loop_speedup', 'gpu_4loop_speedup', 'gpu_4loop_streams_speedup']
        for key in speedup_keys:
            if key not in r:
                r[key] = None
                
        cpu = f"{r['cpu_4loop_speedup']:.2f}x" if r['cpu_4loop_speedup'] is not None else "N/A"
        gpu5 = f"{r['gpu_5loop_speedup']:.2f}x" if r['gpu_5loop_speedup'] is not None else "N/A"
        gpu4 = f"{r['gpu_4loop_speedup']:.2f}x" if r['gpu_4loop_speedup'] is not None else "N/A"
        gpu4s = f"{r['gpu_4loop_streams_speedup']:.2f}x" if r['gpu_4loop_streams_speedup'] is not None else "N/A"
        
        print(f"{r['name']:<25} {cpu:<15} {gpu5:<15} {gpu4:<15} {gpu4s:<15}")
    
    # Create and save the speedup plot
    if args.output:
        plot_speedups(results, baseline_method, contraction_choice, args.output, args.y_max, args.skip, max_runs)
    else:
        plot_speedups(results, baseline_method, contraction_choice, None, args.y_max, args.skip, max_runs)

if __name__ == "__main__":
    main() 