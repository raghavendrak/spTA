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
#    python parse_logs.py TTMC_ncm_2.log -b v4
#
# 5. Parse log file and specify y-axis maximum value:
#    python parse_logs.py TTMC_ncm_2.log -y 15.0
#
# 6. Parse log file and specify baseline method and y-axis maximum value:
#    python parse_logs.py TTMC_ncm_2.log -b v4 -y 15.0
#
# 7. Parse log file and skip specific methods:
#    python parse_logs.py TTMC_ncm_2.log -s v2 v5
#
# 8. Parse log file and perform error analysis:
#    python parse_logs.py TTMC_ncm_2.log -e
#
# 9. Parse log file with ParTI baseline and error analysis:
#    python parse_logs.py TTMC_ncm_2.log -e -b parti
# The script parses TTMC log files containing timing data for tensor contractions
# and generates a bar plot comparing speedups of different GPU implementations
# relative to the specified baseline implementation. It can also perform error
# analysis on validation data to show max absolute diff, relative error, and
# elements with significant error statistics.


import re
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
from tabulate import tabulate

# Method id to default name mapping
DEFAULT_METHOD_NAMES = {
    'v2': 'CPU 4-loop',
    'v3': 'GPU 5-loop',
    'v4': 'GPU 4-loop',
    'v5': 'GPU 4-loop Streams',
    'v6': 'GPU 4-loop ParK',
    'v7': 'GPU 4-loop CM',
    'v8': 'GPU 4-loop WS',
    'v9': 'GPU 4-loop WS v2',
    'v10': 'GPU 4-loop CM v2',
    'parti': 'ParTI TTM'
}

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
    
    # Dictionary to map method IDs to their names
    method_names = {}
    
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
        
        # Find all method runs in this section using regex
        # Updated regex to handle scientific notation (e.g., 1.23e+04)
        method_runs = re.finditer(r"Run \d+/\d+ of method (v\d+)\.\.\..*?Method: (.*?), Time: (\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", 
                                 dataset_section, re.DOTALL)
        
        for method_run in method_runs:
            method_id = method_run.group(1)  # v2, v3, etc.
            method_name = method_run.group(2).split(',')[0].strip()  # Extract method name
            run_time = float(method_run.group(3))  # Correctly parses scientific notation
            
            # Store the mapping from method_id to method_name
            method_names[method_id] = method_name
            
            # Store the timing data
            dataset_timings[base_name][method_id].append(run_time)
    
    # Calculate the maximum number of runs found for any method
    max_runs = 1
    for base_name, timings in dataset_timings.items():
        for method, times in timings.items():
            if len(times) > max_runs:
                max_runs = len(times)
    
    print(f"Detected maximum of {max_runs} runs per method")
    print(f"Detected methods: {', '.join(method_names.keys())}")
    print("Method mappings:")
    for method_id, name in method_names.items():
        print(f"  {method_id}: {name}")
    
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
        
    return results, contraction_choice, max_runs, method_names

def calculate_speedups(results, baseline_method='v5'):
    """Calculate speedup for each method with specified baseline, including error propagation"""
    
    for dataset in results:
        # Skip datasets with no baseline timing or handle them specially
        if baseline_method not in dataset or dataset[baseline_method] is None:
            print(f"Skipping speedup calculation for dataset {dataset['name']} - missing baseline {baseline_method}")
            continue
            
        baseline = dataset[baseline_method]
        baseline_std = dataset.get(f'{baseline_method}_std', 0)
        
        # Get all methods that have timing data for this dataset
        methods = [key for key in dataset.keys() if (key.startswith('v') or key == 'parti') and not key.endswith(('_times', '_std', '_runs', '_cv', '_speedup', '_speedup_std'))]
        
        # Calculate speedups for all methods relative to baseline
        for method in methods:
            if method == baseline_method:
                # The baseline method's speedup is always 1.0
                dataset[f'{method}_speedup'] = 1.0
                dataset[f'{method}_speedup_std'] = 0  # Zero std dev for the baseline
                continue
                
            if dataset[method] is not None:
                dataset[f'{method}_speedup'] = baseline / dataset[method]
                
                # Error propagation for division: relative errors add in quadrature
                if dataset[f'{method}_runs'] > 1 and dataset[f'{baseline_method}_runs'] > 1:
                    rel_err_baseline = baseline_std / baseline if baseline > 0 else 0
                    rel_err_method = dataset[f'{method}_std'] / dataset[method] if dataset[method] > 0 else 0
                    rel_err_speedup = np.sqrt(rel_err_baseline**2 + rel_err_method**2)
                    dataset[f'{method}_speedup_std'] = dataset[f'{method}_speedup'] * rel_err_speedup
                else:
                    dataset[f'{method}_speedup_std'] = 0
            else:
                dataset[f'{method}_speedup'] = None
                dataset[f'{method}_speedup_std'] = 0
            
    return results

def format_time(time_value):
    """Format time value with proper units and significant digits"""
    if time_value is None:
        return "N/A"
    
    if time_value >= 1e6:
        return f"{time_value/1e6:.2f}x10⁶ s"
    elif time_value >= 1e3:
        return f"{time_value/1e3:.2f}s"
    else:
        return f"{time_value:.2f}ms"

def plot_speedups(results, baseline_method='v5', method_names=None, contraction_choice=None, 
                 output_file=None, y_max=5, skip_methods=None, runs_per_method=1):
    """Create a bar plot showing speedups for all datasets
    
    Parameters:
    -----------
    results : list
        List of dictionaries containing timing and speedup data for each dataset
    baseline_method : str
        Method used as baseline for speedup calculations (e.g. 'v4')
    method_names : dict
        Dictionary mapping method IDs to human-readable names
    contraction_choice : str
        Contraction operation used in the benchmark
    output_file : str
        Path to save the output plot image
    y_max : float
        Maximum value for y-axis
    skip_methods : list
        List of method IDs to skip in the plot (e.g., ['v2', 'v5'])
    runs_per_method : int
        Number of runs performed for each method (used in plot title)
    """
    
    # If method_names not provided, use default mapping
    if method_names is None:
        method_names = DEFAULT_METHOD_NAMES
    
    # Initialize skip_methods if not provided
    if skip_methods is None:
        skip_methods = []
    
    # Filter results to only include datasets with the baseline method available
    filtered_results = [r for r in results if baseline_method in r and r[baseline_method] is not None]
    
    if not filtered_results:
        print("No datasets have the baseline method available for plotting")
        return
    
    # Find all unique method IDs across all datasets
    all_method_ids = set()
    for r in filtered_results:
        for key in r.keys():
            if key.startswith('v') and not key.endswith(('_times', '_std', '_runs', '_cv', '_speedup', '_speedup_std')):
                all_method_ids.add(key)
    
    # Sort method IDs by version number
    all_method_ids = sorted(all_method_ids, key=lambda x: int(x[1:]))
    
    # Filter out methods that should be skipped
    method_ids = [method for method in all_method_ids if method not in skip_methods and method != baseline_method]
    # Always include baseline method even if it's in skip_methods
    method_ids.append(baseline_method)
    # Remove duplicates and maintain order
    method_ids = sorted(set(method_ids), key=method_ids.index)
    
    # Extract data for plotting (only for methods we're not skipping)
    datasets = [r['name'] for r in filtered_results]
    
    # Extract speedups and standard deviations for methods we want to plot
    speedup_data = []
    std_devs = []  # Standard deviations for each method
    
    for method in method_ids:
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
    
    # Create user-friendly method names for the plot
    display_names = {}
    for method_id in method_ids:
        if method_id in method_names:
            display_names[method_id] = method_names[method_id]
        else:
            display_names[method_id] = f"{method_id}"
    
    baseline_label = display_names[baseline_method] + ' (Baseline)'
    
    # Setup figure and axis
    fig, ax = plt.subplots(figsize=(18, 6))
    
    # Set up bar positions
    x = np.arange(len(datasets))
    bar_width = 0.1
    
    # Create bars with appropriate labels
    bar_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Calculate positions based on number of methods being plotted
    num_methods = len(method_ids)
    if num_methods <= 1:
        bar_positions = [0]  # Only baseline
    else:
        # Calculate positions to center the bars
        total_width = bar_width * (num_methods - 1)
        start_pos = -total_width / 2
        bar_positions = [start_pos + i * bar_width for i in range(num_methods)]
    
    # Create bars for each method with error bars for standard deviation
    for i, (method, data, std, pos) in enumerate(zip(method_ids, speedup_data, std_devs, bar_positions)):
        label = display_names[method]
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
        for j, (method, data, std, pos) in enumerate(zip(method_ids, speedup_data, std_devs, bar_positions)):
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
            else:
                ax.text(x[i] + pos, 0.01, 
                        "O.O.M", ha='center', va='bottom', fontsize=8, 
                        rotation=90)

    
    # Customize plot
    ax.set_xlabel('Dataset')
    ax.set_ylabel(f'Speedup (relative to {display_names[baseline_method]})')
    
    # Set the title with contraction choice and run info if available
    title = f'TTMC Method Performance Comparison (Baseline: {display_names[baseline_method]})'
    if runs_per_method > 1:
        title += f' - Averaged from {runs_per_method} runs'
    if contraction_choice:
        title += f'\nContraction: {contraction_choice}'
    ax.set_title(title)
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Add memory error notations
    for i, dataset in enumerate(filtered_results):
        positions = bar_positions
        
        for method_idx, method in enumerate(method_ids):
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

def parse_validation_data(log_file_path):
    """Parse validation data from the TTMC log file and extract error metrics by method"""
    
    validation_results = []
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Dictionary to store validation information by dataset name
    dataset_validations = defaultdict(lambda: defaultdict(list))
    
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
        
        # Find method runs and their associated validation data
        method_runs = re.finditer(r"Run \d+/\d+ of method (v\d+)\.\.\..*?Method: (.*?), Time: (\d+(?:\.\d+)?(?:[eE][+-]?\d+)?).*?Validation: Max absolute diff = ([\d\.e\-\+]+), Relative error = ([\d\.e\-\+]+), Elements with significant error = ([\d]+)", 
                                 dataset_section, re.DOTALL)
        
        for method_run in method_runs:
            method_id = method_run.group(1)  # v2, v3, etc.
            max_abs_diff = float(method_run.group(4))
            relative_error = float(method_run.group(5))
            significant_error_elements = int(method_run.group(6))
            
            dataset_validations[base_name][method_id].append({
                'max_abs_diff': max_abs_diff,
                'relative_error': relative_error,
                'significant_error_elements': significant_error_elements
            })
    
    # Convert to results format similar to timing results
    for dataset, methods in dataset_validations.items():
        result = {'name': dataset}
        
        for method_id, validations in methods.items():
            if validations:
                # Calculate averages for each metric
                avg_max_abs_diff = np.mean([v['max_abs_diff'] for v in validations])
                avg_relative_error = np.mean([v['relative_error'] for v in validations])
                avg_sig_error_elements = np.mean([v['significant_error_elements'] for v in validations])
                
                result[f"{method_id}_max_abs_diff"] = avg_max_abs_diff
                result[f"{method_id}_relative_error"] = avg_relative_error
                result[f"{method_id}_sig_error_elements"] = avg_sig_error_elements
        
        validation_results.append(result)
    
    return validation_results

def print_error_tables(validation_results, method_names):
    """Print formatted error tables with datasets as rows and methods as columns"""
    
    if not validation_results:
        print("No validation data found.")
        return
    
    # Sort results by dataset name
    validation_results.sort(key=lambda x: x['name'])
    
    # Get all unique methods across datasets
    all_methods = set()
    for r in validation_results:
        for key in r.keys():
            if key.startswith('v') and key.endswith('_max_abs_diff'):
                method_id = key.replace('_max_abs_diff', '')
                all_methods.add(method_id)
    
    # Sort methods by version number
    all_methods = sorted(all_methods, key=lambda x: int(x[1:]))
    
    print("\n" + "="*120)
    print("ERROR ANALYSIS SUMMARY")
    print("="*120)
    
    # Table 1: Max Absolute Difference
    print("\nTable 1: Maximum Absolute Difference")
    print("-" * 80)
    
    max_abs_diff_data = []
    headers = ['Dataset'] + [f"{method} ({method_names.get(method, 'Unknown')})" for method in all_methods]
    
    for r in validation_results:
        row = [r['name']]
        for method in all_methods:
            key = f"{method}_max_abs_diff"
            if key in r and r[key] is not None:
                row.append(f"{r[key]:.6e}")
            else:
                row.append('N/A')
        max_abs_diff_data.append(row)
    
    print(tabulate(max_abs_diff_data, headers=headers, tablefmt='grid'))
    
    # Table 2: Relative Error
    print("\nTable 2: Relative Error")
    print("-" * 80)
    
    rel_error_data = []
    for r in validation_results:
        row = [r['name']]
        for method in all_methods:
            key = f"{method}_relative_error"
            if key in r and r[key] is not None:
                row.append(f"{r[key]:.6e}")
            else:
                row.append('N/A')
        rel_error_data.append(row)
    
    print(tabulate(rel_error_data, headers=headers, tablefmt='grid'))
    
    # Table 3: Elements with Significant Error
    print("\nTable 3: Elements with Significant Error")
    print("-" * 80)
    
    sig_error_data = []
    for r in validation_results:
        row = [r['name']]
        for method in all_methods:
            key = f"{method}_sig_error_elements"
            if key in r and r[key] is not None:
                row.append(f"{r[key]:,.0f}")
            else:
                row.append('N/A')
        sig_error_data.append(row)
    
    print(tabulate(sig_error_data, headers=headers, tablefmt='grid'))

def parse_ttm_baseline_results(ttm_results_file="/home/bhaskar/spTA/Week_11_TTMC_GPU/ttm_results.txt"):
    """
    Parse TTM baseline results from the benchmark results file.
    
    Parameters:
    -----------
    ttm_results_file : str
        Path to the TTM results file (default: /home/bhaskar/spTA/Week_11_TTMC_GPU/ttm_results.txt)
    
    Returns:
    --------
    dict
        Dictionary mapping dataset names to TTM kernel times in seconds
    """
    ttm_results = {}
    
    try:
        with open(ttm_results_file, 'r') as f:
            lines = f.readlines()
        
        # Skip header line
        for line in lines[1:]:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    dataset = parts[0].strip()
                    time_str = parts[1].strip()
                    
                    if time_str != 'ERROR':
                        try:
                            # Convert to milliseconds to match TTMC log format
                            time_val = float(time_str) * 1000  # Convert seconds to milliseconds
                            time_val = time_val 
                            ttm_results[dataset] = time_val
                        except ValueError:
                            print(f"Warning: Could not parse time value '{time_str}' for dataset '{dataset}'")
                    else:
                        print(f"Warning: TTM benchmark failed for dataset '{dataset}'")
        
        print(f"Loaded TTM baseline results for {len(ttm_results)} datasets")
        return ttm_results
    
    except FileNotFoundError:
        print(f"Warning: TTM results file not found: {ttm_results_file}")
        return {}
    except Exception as e:
        print(f"Error reading TTM results file: {e}")
        return {}

def integrate_ttm_baseline(results, ttm_results, method_names):
    """
    Integrate TTM baseline results into the main results data structure.
    
    Parameters:
    -----------
    results : list
        List of dictionaries containing timing data for each dataset
    ttm_results : dict
        Dictionary mapping dataset names to TTM kernel times
    
    Returns:
    --------
    list
        Updated results with TTM baseline data integrated
    """
    for r in results:
        dataset_name = r['name']
        if dataset_name in ttm_results:
            # Add TTM baseline as 'v11 parti' method
            r['parti'] = ttm_results[dataset_name]
            r['parti_times'] = [ttm_results[dataset_name]]  # Single measurement
            r['parti_std'] = 0.0  # No standard deviation for single measurement
            r['parti_runs'] = 1
            r['parti_cv'] = 0.0  # Coefficient of variation is 0 for single measurement
    method_names['parti'] = 'PARTI'
    return results, method_names

def main():
    parser = argparse.ArgumentParser(description='Parse TTMC log files and create performance plots')
    parser.add_argument('log_file', help='Path to the TTMC log file')
    parser.add_argument('-o', '--output', help='Output file path for the plot (e.g., speedup_plot.png)')
    parser.add_argument('-b', '--baseline', default='v5', 
                        help='Method to use as the baseline for speedup calculation (e.g., v2, v3, v4)')
    parser.add_argument('-y', '--y-max', type=float, default=5.0,
                        help='Maximum value for the y-axis (default: 5.0)')
    parser.add_argument('-s', '--skip', nargs='+',
                        help='Methods to skip in the plot (e.g., -s v2 v6)')
    parser.add_argument('-e', '--error-analysis', action='store_true',
                        help='Perform error analysis and display error tables')
    args = parser.parse_args()
    
    # Parse log file
    print(f"Parsing log file: {args.log_file}")
    results, contraction_choice, max_runs, method_names = parse_log_file(args.log_file)
    
    if not results:
        print("No valid results found in log file.")
        return
    
    if contraction_choice:
        print(f"Found contraction choice: {contraction_choice}")
    
    # Handle TTM baseline integration if parti is specified as baseline
    baseline_method = args.baseline
    if baseline_method == 'parti':
        print("Loading TTM baseline results from ParTI benchmark...")
        ttm_results = parse_ttm_baseline_results()
        if ttm_results:
            # Integrate TTM baseline results into main results
            results, method_names = integrate_ttm_baseline(results, ttm_results, method_names)
            print(f"Integrated TTM baseline for {len([r for r in results if 'parti' in r])} datasets")
        else:
            print("Warning: No TTM baseline results found. Cannot use 'parti' as baseline.")
            print("Please run the TTM benchmark first or choose a different baseline method.")
            return
    
    # Calculate speedups
    results = calculate_speedups(results, baseline_method)
    
    # Print summary
    print(f"\nUsing {baseline_method} ({method_names.get(baseline_method, 'Unknown')}) as baseline")
    
    if args.skip:
        print(f"Skipping methods: {', '.join(args.skip)}")
    
    # Get all unique methods across datasets
    all_methods = set()
    for r in results:
        for key in r.keys():
            if key.startswith('v') and not key.endswith(('_times', '_std', '_runs', '_cv', '_speedup', '_speedup_std')):
                all_methods.add(key)
    
    if baseline_method == 'parti':
        all_methods.add('parti')
    # Sort methods by version number
    all_methods = sorted(all_methods, key=lambda x: int(x[1:]) if x[1:].isdigit() else float('inf'))
    
    print("\nPerformance Summary (Average Times):")
    print("-" * 100)
    
    # Prepare data for tabulate
    performance_data = []
    headers = ['Dataset'] + [f"{method} ({method_names.get(method, 'Unknown')})" for method in all_methods] + ['Best Method']
    
    for r in results:
        row = [r['name']]
        min_time = float('inf')
        best_method = "N/A"
        best_method_time = None

        # First pass to find best method and its time
        for method in all_methods:
            if method in r and r[method] is not None and r[method] < min_time:
                min_time = r[method]
                best_method = f"{method} ({method_names.get(method, 'Unknown')})"
                best_method_time = r[method]

        # Second pass to format times and highlight best method
        for method in all_methods:
            if method in r and r[method] is not None:
                time_val = r[method]
                # Format time with proper units
                if time_val >= 1e6:
                    time_str = f"{time_val / 1e6:.2f}s (x10⁶)"
                elif time_val >= 1e3:
                    time_str = f"{time_val / 1e3:.2f}s"
                else:
                    time_str = f"{time_val:.2f}ms"

                # Highlight the best method's time
                if time_val == best_method_time:
                    time_str = f"*{time_str}*"
                row.append(time_str)
            else:
                row.append('N/A')

        row.append(best_method)
        performance_data.append(row)
    
    print(tabulate(performance_data, headers=headers, tablefmt='grid'))
    
    print("\nSpeedup Summary (relative to {}: {}):".format(baseline_method, method_names.get(baseline_method, 'Unknown')))
    print("-" * 100)
    
    # Prepare data for speedup tabulate
    speedup_data = []
    speedup_headers = ['Dataset'] + [f"{method} ({method_names.get(method, 'Unknown')})" for method in all_methods]
    
    for r in results:
        # Skip datasets with no baseline timing
        if baseline_method not in r or r[baseline_method] is None:
            continue
            
        row = [r['name']]
        for method in all_methods:
            speedup_key = f"{method}_speedup"
            if speedup_key in r and r[speedup_key] is not None:
                row.append(f"{r[speedup_key]:.2f}x")
            else:
                row.append('N/A')
        speedup_data.append(row)
    
    print(tabulate(speedup_data, headers=speedup_headers, tablefmt='grid'))
    
    # Perform error analysis if requested
    if args.error_analysis:
        print("\nPerforming error analysis...")
        validation_results = parse_validation_data(args.log_file)
        if validation_results:
            print_error_tables(validation_results, method_names)
        else:
            print("No validation data found in log file.")
    
    # Create and save the speedup plot
    plot_speedups(results, baseline_method, method_names, contraction_choice, 
                 args.output, args.y_max, args.skip, max_runs)

if __name__ == "__main__":
    main() 