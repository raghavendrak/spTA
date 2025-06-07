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
        
    # Extract contraction choice from the first dataset section
    contraction_match = re.search(r"Your Contraction Choice : (.*?)\n", content)
    if contraction_match:
        contraction_choice = contraction_match.group(1)
        
    # Split the content by dataset sections
    dataset_sections = re.split(r"Contraction results for (.*?)\n", content)[1:]
    
    # Process each dataset section
    for i in range(0, len(dataset_sections), 2):
        if i+1 >= len(dataset_sections):
            break
            
        dataset_name = dataset_sections[i].strip()
        section_content = dataset_sections[i+1]
        
        # Extract base dataset name without path and extension
        base_name = os.path.basename(dataset_name)
        base_name = os.path.splitext(base_name)[0]
        
        # Initialize timing dict for this dataset
        dataset_result = {
            'name': base_name,
            'cpu_4loop': None,
            'gpu_5loop': None,  # baseline
            'gpu_4loop': None,
            'gpu_4loop_streams': None
        }
        
        # Extract timing information using regex patterns
        cpu_4loop_match = re.search(r"Time taken by CPU Method - 2 \[4-for loop\] i.e. contraction 2 : (\d+\.\d+)", section_content)
        if cpu_4loop_match:
            dataset_result['cpu_4loop'] = float(cpu_4loop_match.group(1))
            
        gpu_5loop_match = re.search(r"Time taken by GPU Method - 1 \[5-for loop\] i.e. contraction 3 : (\d+\.\d+)", section_content)
        if gpu_5loop_match:
            dataset_result['gpu_5loop'] = float(gpu_5loop_match.group(1))
            
        gpu_4loop_match = re.search(r"Time taken by GPU Method - 2 \[4-for loop\] i.e. contraction 4 : (\d+\.\d+)", section_content)
        if gpu_4loop_match:
            dataset_result['gpu_4loop'] = float(gpu_4loop_match.group(1))
            
        gpu_4loop_streams_match = re.search(r"Time taken by GPU Method - 3 \[4-for loop\] i.e. streams: (\d+\.\d+)", section_content)
        if gpu_4loop_streams_match:
            dataset_result['gpu_4loop_streams'] = float(gpu_4loop_streams_match.group(1))
            
        if (dataset_result['cpu_4loop'] is None) and (dataset_result['gpu_5loop'] is None) and (dataset_result['gpu_4loop'] is None) and (dataset_result['gpu_4loop_streams'] is None):
            print(f"Skipping dataset {base_name} because it has no timing data")
            continue
        
        results.append(dataset_result)
        
    return results, contraction_choice

def calculate_speedups(results, baseline_method='gpu_4loop'):
    """Calculate speedup for each method with specified baseline"""
    
    for dataset in results:
        # Skip datasets with no baseline timing
        if dataset[baseline_method] is None:
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

def plot_speedups(results, baseline_method='gpu_4loop', contraction_choice=None, output_file=None, y_max=10):
    """Create a bar plot showing speedups for all datasets"""
    
    # Extract data for plotting
    datasets = [r['name'] for r in results]
    cpu_4loop_speedups = [r.get('cpu_4loop_speedup', np.nan) for r in results]
    gpu_5loop_speedups = [r.get('gpu_5loop_speedup', np.nan) for r in results]
    gpu_4loop_speedups = [r.get('gpu_4loop_speedup', np.nan) for r in results]
    gpu_4loop_streams_speedups = [r.get('gpu_4loop_streams_speedup', np.nan) for r in results]
    
    # Convert any None to NaN for plotting
    cpu_4loop_speedups = [s if s is not None else np.nan for s in cpu_4loop_speedups]
    gpu_5loop_speedups = [s if s is not None else np.nan for s in gpu_5loop_speedups]
    gpu_4loop_speedups = [s if s is not None else np.nan for s in gpu_4loop_speedups]
    gpu_4loop_streams_speedups = [s if s is not None else np.nan for s in gpu_4loop_streams_speedups]
    
    # Get the method labels
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
    bar_positions = [-bar_width*1.5, -bar_width*0.5, bar_width*0.5, bar_width*1.5]
    speedup_data = [cpu_4loop_speedups, gpu_5loop_speedups, gpu_4loop_speedups, gpu_4loop_streams_speedups]
    method_keys = ['cpu_4loop', 'gpu_5loop', 'gpu_4loop', 'gpu_4loop_streams']
    
    for i, (method, data, pos, color) in enumerate(zip(method_keys, speedup_data, bar_positions, bar_colors)):
        label = method_names[method]
        if method == baseline_method:
            label = baseline_label
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
    for i, dataset in enumerate(results):
        methods = ['cpu_4loop_speedup', 'gpu_5loop_speedup', 'gpu_4loop_speedup', 'gpu_4loop_streams_speedup']
        positions = [i - bar_width*1.5, i - bar_width*0.5, i + bar_width*0.5, i + bar_width*1.5]
        
        for method, pos in zip(methods, positions):
            if method in dataset and dataset[method] is None:
                ax.text(pos, 0.05, 'Mem Error', ha='center', va='bottom', rotation=90, 
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
    
    # Print speedup summary
    print(f"\nSpeedup Summary (relative to {method_names[baseline_method]}):")
    print("-" * 100)
    print(f"{'Dataset':<25} {'CPU 4-loop':<15} {'GPU 5-loop':<15} {'GPU 4-loop':<15} {'GPU 4-loop Strm':<15}")
    print("-" * 100)
    
    for r in results:
        cpu = f"{r['cpu_4loop_speedup']:.2f}x" if r['cpu_4loop_speedup'] is not None else "N/A"
        gpu5 = f"{r['gpu_5loop_speedup']:.2f}x" if r['gpu_5loop_speedup'] is not None else "N/A"
        gpu4 = f"{r['gpu_4loop_speedup']:.2f}x" if r['gpu_4loop_speedup'] is not None else "N/A"
        gpu4s = f"{r['gpu_4loop_streams_speedup']:.2f}x" if r['gpu_4loop_streams_speedup'] is not None else "N/A"
        
        print(f"{r['name']:<25} {cpu:<15} {gpu5:<15} {gpu4:<15} {gpu4s:<15}")
    
    # Create and save/show the plot
    plot_speedups(results, baseline_method, contraction_choice, args.output, args.y_max)

if __name__ == "__main__":
    main() 