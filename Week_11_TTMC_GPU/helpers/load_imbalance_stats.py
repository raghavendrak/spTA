#!/usr/bin/env python3
"""
Load Imbalance Statistics Parser for TTMC GPU Log Files

This script parses TTMC_ncm_0.log and extracts load imbalance statistics
for each dataset, including mode-wise min, max, and stddev values.
"""

import re
import csv
from typing import Dict, List, Tuple, Optional

def parse_log_file(log_file_path: str) -> List[Dict]:
    """
    Parse the TTMC log file and extract load imbalance statistics.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        List of dictionaries containing dataset statistics
    """
    datasets = []
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Split content by dataset sections (separated by dashes)
    sections = content.split('---------------------------------')
    
    for section in sections:
        lines = section.strip().split('\n')
        if not lines or 'Running contraction on' not in lines[0]:
            continue
            
        dataset_info = {}
        
        # Extract dataset name
        dataset_match = re.search(r'Running contraction on (\./.*?\.csf)\.\.\.', lines[0])
        if dataset_match:
            dataset_name = dataset_match.group(1).replace('./', '').replace('.csf', '')
            dataset_info['dataset_name'] = dataset_name
        else:
            continue
            
        # Extract tensor order
        for line in lines:
            if 'Found tensor order:' in line:
                order_match = re.search(r'Found tensor order: (\d+)', line)
                if order_match:
                    dataset_info['tensor_order'] = int(order_match.group(1))
                break
                
        # Extract mode statistics
        mode_stats = {}
        for line in lines:
            if 'Mode' in line and 'nnz_length:' in line:
                mode_match = re.search(r'Mode (\d+) nnz_length: min=([\d.]+), max=([\d.]+), stddev=([\d.]+)', line)
                if mode_match:
                    mode_num = int(mode_match.group(1))
                    min_val = float(mode_match.group(2))
                    max_val = float(mode_match.group(3))
                    stddev_val = float(mode_match.group(4))
                    
                    mode_stats[f'mode_{mode_num}_min'] = min_val
                    mode_stats[f'mode_{mode_num}_max'] = max_val
                    mode_stats[f'mode_{mode_num}_stddev'] = stddev_val
        
        # Only add if we have valid data
        if 'dataset_name' in dataset_info and 'tensor_order' in dataset_info and mode_stats:
            dataset_info.update(mode_stats)
            datasets.append(dataset_info)
    
    return datasets

def get_all_columns(datasets: List[Dict]) -> List[str]:
    """
    Get all column names from the datasets.
    
    Args:
        datasets: List of dataset dictionaries
        
    Returns:
        List of column names
    """
    if not datasets:
        return []
    
    # Start with base columns
    columns = ['dataset_name', 'tensor_order']
    
    # Find all mode columns
    mode_columns = set()
    for dataset in datasets:
        for key in dataset.keys():
            if key.startswith('mode_'):
                mode_columns.add(key)
    
    # Sort mode columns by mode number and statistic type
    mode_columns = sorted(mode_columns, key=lambda x: (
        int(x.split('_')[1]),  # mode number
        x.split('_')[2]        # statistic type (min, max, stddev)
    ))
    
    return columns + list(mode_columns)

def print_tabulated_results(datasets: List[Dict]):
    """
    Print the results in a nicely formatted table.
    
    Args:
        datasets: List of dataset dictionaries
    """
    if not datasets:
        print("No data found in log file.")
        return
    
    print("=" * 120)
    print("LOAD IMBALANCE STATISTICS FROM TTMC GPU LOG")
    print("=" * 120)
    print()
    
    # Get all columns
    columns = get_all_columns(datasets)
    
    # Print summary statistics
    tensor_orders = set(dataset['tensor_order'] for dataset in datasets)
    print(f"Total datasets analyzed: {len(datasets)}")
    print(f"Tensor orders found: {sorted(tensor_orders)}")
    print()
    
    # Print header
    header = "Dataset Name".ljust(35) + "Order".ljust(8)
    for col in columns[2:]:  # Skip dataset_name and tensor_order
        if 'min' in col:
            mode_num = col.split('_')[1]
            header += f"Mode{mode_num}_min".ljust(12)
        elif 'max' in col:
            mode_num = col.split('_')[1]
            header += f"Mode{mode_num}_max".ljust(12)
        elif 'stddev' in col:
            mode_num = col.split('_')[1]
            header += f"Mode{mode_num}_std".ljust(12)
    
    print(header)
    print("-" * len(header))
    
    # Print data rows
    for dataset in datasets:
        row = dataset['dataset_name'].ljust(35) + str(dataset['tensor_order']).ljust(8)
        for col in columns[2:]:
            if col in dataset:
                row += f"{dataset[col]:.3f}".ljust(12)
            else:
                row += "N/A".ljust(12)
        print(row)
    
    print()
    
    # Print statistics by tensor order
    print("=" * 120)
    print("SUMMARY BY TENSOR ORDER")
    print("=" * 120)
    
    for order in sorted(tensor_orders):
        order_datasets = [d for d in datasets if d['tensor_order'] == order]
        print(f"\nTensor Order {order} ({len(order_datasets)} datasets):")
        print("-" * 80)
        
        for dataset in order_datasets:
            print(f"Dataset: {dataset['dataset_name']}")
            for mode in range(1, order):
                min_col = f'mode_{mode}_min'
                max_col = f'mode_{mode}_max'
                stddev_col = f'mode_{mode}_stddev'
                
                if min_col in dataset and max_col in dataset and stddev_col in dataset:
                    print(f"  Mode {mode}: min={dataset[min_col]:.3f}, max={dataset[max_col]:.3f}, stddev={dataset[stddev_col]:.3f}")
            print()

def save_to_csv(datasets: List[Dict], output_file: str = 'load_imbalance_stats.csv'):
    """
    Save the results to a CSV file.
    
    Args:
        datasets: List of dataset dictionaries
        output_file: Output CSV file path
    """
    if not datasets:
        print("No data to save.")
        return
    
    columns = get_all_columns(datasets)
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(datasets)
    
    print(f"Results saved to {output_file}")

def main():
    """Main function to parse log file and display results."""
    log_file = 'TTMC_ncm_0.log'
    
    try:
        # Parse the log file
        print(f"Parsing log file: {log_file}")
        datasets = parse_log_file(log_file)
        
        if not datasets:
            print("No valid dataset entries found in the log file.")
            return
        
        # Display results
        print_tabulated_results(datasets)
        
        # Save to CSV
        save_to_csv(datasets)
        
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found.")
        print("Please ensure the log file is in the current directory.")
    except Exception as e:
        print(f"Error parsing log file: {e}")

if __name__ == "__main__":
    main()
