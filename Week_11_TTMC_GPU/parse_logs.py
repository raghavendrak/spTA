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

def parse_log_file(log_file_path):
    """Parse the TTMC log file and extract timing data for each dataset"""
    
    results = []
    current_dataset = None
    contraction_choice = None
    
    with open(log_file_path, 'r') as f:
        content = f.read()
        
    # Extract contraction choice from the beginning of the file
    contraction_match = re.search(r"Your Contraction Choice : (.*?)\n", content)
    if contraction_match:
        contraction_choice = contraction_match.group(1)
        
    # Split the content by separator lines
    sections = re.split(r"---------------------------------\n", content)
    
    # Dictionary to store timing information by dataset name
    dataset_timings = {}
    
    # Process each section
    for section in sections:
        if not section.strip():
            continue
            
        # Extract dataset name
        dataset_match = re.search(r"Running contraction on (.*?)\.\.\.?", section)
        if not dataset_match:
            continue
            
        dataset_name = dataset_match.group(1).strip()
        
        # Extract base dataset name without path and extension
        base_name = os.path.basename(dataset_name)
        base_name = os.path.splitext(base_name)[0]
        
        # Initialize dataset result if first time seeing this dataset
        if base_name not in dataset_timings:
            dataset_timings[base_name] = {
                'name': base_name,
                'cpu_4loop': None,
                'gpu_5loop': None,
                'gpu_4loop': None,
                'gpu_4loop_streams': None
            }
        
        # Check for CPU reference time
        cpu_4loop_match = re.search(r"Method: CPU 4-loop \(reference\), Time: (\d+(?:\.\d+)?) ms", section)
        if cpu_4loop_match:
            dataset_timings[base_name]['cpu_4loop'] = float(cpu_4loop_match.group(1))
        
        # Check for GPU 5-loop time
        gpu_5loop_match = re.search(r"Method: GPU 5-loop, Time: (\d+(?:\.\d+)?) ms", section)
        if gpu_5loop_match:
            dataset_timings[base_name]['gpu_5loop'] = float(gpu_5loop_match.group(1))
        
        # Check for GPU 4-loop time
        gpu_4loop_match = re.search(r"Method: GPU 4-loop, Time: (\d+(?:\.\d+)?) ms", section)
        if gpu_4loop_match:
            dataset_timings[base_name]['gpu_4loop'] = float(gpu_4loop_match.group(1))
        
        # Check for GPU 4-loop streams time
        gpu_4loop_streams_match = re.search(r"Method: GPU 4-loop streams, Time: (\d+(?:\.\d+)?) ms", section)
        if gpu_4loop_streams_match:
            dataset_timings[base_name]['gpu_4loop_streams'] = float(gpu_4loop_streams_match.group(1))
    
    # Convert dictionary to list
    for base_name, timings in dataset_timings.items():
        if (timings['cpu_4loop'] is None and 
            timings['gpu_5loop'] is None and 
            timings['gpu_4loop'] is None and 
            timings['gpu_4loop_streams'] is None):
            print(f"Skipping dataset {base_name} because it has no timing data")
            continue
            
        results.append(timings)
        
    return results, contraction_choice

def calculate_speedups(results, baseline_method='gpu_4loop'):
    """Calculate speedup for each method with specified baseline"""
    
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
        
        # Calculate speedups (baseline / method_time)
        if dataset['cpu_4loop'] is not None:
            dataset['cpu_4loop_speedup'] = baseline / dataset['cpu_4loop']
        else:
            dataset['cpu_4loop_speedup'] = None
            
        if dataset['gpu_5loop'] is not None:
            dataset['gpu_5loop_speedup'] = baseline / dataset['gpu_5loop']
        else:
            dataset['gpu_5loop_speedup'] = None
        
        # The baseline method's speedup is always 1.0
        dataset[f'{baseline_method}_speedup'] = 1.0
            
        if baseline_method != 'gpu_4loop' and dataset['gpu_4loop'] is not None:
            dataset['gpu_4loop_speedup'] = baseline / dataset['gpu_4loop']
        else:
            if baseline_method != 'gpu_4loop':
                dataset['gpu_4loop_speedup'] = None
            
        if baseline_method != 'gpu_4loop_streams' and dataset['gpu_4loop_streams'] is not None:
            dataset['gpu_4loop_streams_speedup'] = baseline / dataset['gpu_4loop_streams']
        else:
            if baseline_method != 'gpu_4loop_streams':
                dataset['gpu_4loop_streams_speedup'] = None
            
    return results

def plot_speedups(results, baseline_method='gpu_4loop', contraction_choice=None, output_file=None, y_max=10, skip_methods=None):
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
    
    # Extract speedups only for methods we want to plot
    speedup_data = []
    for method in method_keys:
        speedup_key = f"{method}_speedup"
        speedups = [r.get(speedup_key, np.nan) for r in filtered_results]
        speedups = [s if s is not None else np.nan for s in speedups]
        speedup_data.append(speedups)
    
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
    
    # Create bars for each method
    for i, (method, data, pos) in enumerate(zip(method_keys, speedup_data, bar_positions)):
        label = method_names[method]
        if method == baseline_method:
            label = baseline_label
        color = bar_colors[i % len(bar_colors)]  # Cycle through colors if needed
        ax.bar(x + pos, data, bar_width, label=label, color=color)
    
    # Add horizontal line at y=1
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    
    # Add speedup values on top of bars
    for i in range(len(datasets)):
        # Add labels for all methods
        for j, (method, data, pos) in enumerate(zip(method_keys, speedup_data, bar_positions)):
            if not np.isnan(data[i]):
                height = data[i]
                if height > y_max - 1:  # Adjust position for very tall bars
                    # Place label at 90% of y-axis limit
                    text_height = y_max * 0.9
                    ax.text(x[i] + pos, text_height, 
                            f"{height:.2f}x", ha='center', va='bottom', fontsize=8, 
                            rotation=75, color='black', weight='bold')
                else:
                    ax.text(x[i] + pos, height + 0.05*height, 
                            f"{height:.2f}x", ha='center', va='bottom', fontsize=8, 
                            rotation=75)
    
    # Customize plot
    ax.set_xlabel('Dataset')
    ax.set_ylabel(f'Speedup (relative to {method_names[baseline_method]})')
    
    # Set the title with contraction choice if available
    title = f'TTMC Method Performance Comparison (Baseline: {method_names[baseline_method]})'
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
    parser.add_argument('-y', '--y-max', type=float, default=10.0,
                        help='Maximum value for the y-axis (default: 10.0)')
    parser.add_argument('-s', '--skip', nargs='+', choices=['cpu_4loop', 'gpu_5loop', 'gpu_4loop', 'gpu_4loop_streams'],
                        help='Methods to skip in the plot (e.g., -s cpu_4loop gpu_5loop)')
    args = parser.parse_args()
    
    # Parse log file
    print(f"Parsing log file: {args.log_file}")
    results, contraction_choice = parse_log_file(args.log_file)
    
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
    
    print("\nPerformance Summary:")
    print("-" * 100)
    print(f"{'Dataset':<25} {'CPU 4-loop':<15} {'GPU 5-loop':<15} {'GPU 4-loop':<15} {'GPU 4-loop Strm':<15}")
    print("-" * 100)
    
    for r in results:
        cpu = f"{r['cpu_4loop']:.2f}ms" if r['cpu_4loop'] is not None else "N/A"
        gpu5 = f"{r['gpu_5loop']:.2f}ms" if r['gpu_5loop'] is not None else "N/A"
        gpu4 = f"{r['gpu_4loop']:.2f}ms" if r['gpu_4loop'] is not None else "N/A"
        gpu4s = f"{r['gpu_4loop_streams']:.2f}ms" if r['gpu_4loop_streams'] is not None else "N/A"
        
        print(f"{r['name']:<25} {cpu:<15} {gpu5:<15} {gpu4:<15} {gpu4s:<15}")
    
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
        plot_speedups(results, baseline_method, contraction_choice, args.output, args.y_max, args.skip)
    else:
        plot_speedups(results, baseline_method, contraction_choice, None, args.y_max, args.skip)

if __name__ == "__main__":
    main() 