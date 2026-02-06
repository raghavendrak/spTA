#!/usr/bin/env python3
"""
Script to extract tensor data from B-CSF log file and output as CSV.

This script parses the bcsf_all_datasets.log file and extracts key tensor information
for each dataset, outputting the results in CSV format.
"""

import re
import sys
import os

def extract_tensor_data(log_file_path):
    """
    Extract tensor data from the B-CSF log file.
    
    Returns:
        list: List of dictionaries containing tensor data
    """
    tensor_data = []
    
    try:
        with open(log_file_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found.")
        return []
    
    # Split content into sections for each dataset
    dataset_sections = re.split(r'Progress: \d+/\d+\nProcessing dataset: ', content)
    
    for section in dataset_sections[1:]:  # Skip the first empty section
        try:
            # Extract dataset name
            dataset_match = re.search(r'^([^\n]+)', section)
            if not dataset_match:
                continue
            dataset_name = dataset_match.group(1).strip()
            
            # Skip if there's a CUDA runtime error or other major error
            if 'CUDA Runtime Error' in section or 'Assertion' in section:
                tensor_data.append({
                    'dataset': dataset_name,
                    'dimensions': 'ERROR',
                    'total_nnz': 'ERROR',
                    'execution_time_ms': 'ERROR',
                    'correctness': 'ERROR',
                    'tile_time_ms': 'ERROR',
                    'tile_nnz': 'ERROR',
                    'tile_nfibers': 'ERROR',
                    'tile_nslc': 'ERROR'
                })
                continue
            
            # Extract tensor dimensions
            dim_match = re.search(r'Tensor dimensions: (\d+) x (\d+) x (\d+)', section)
            if dim_match:
                dimensions = f"{dim_match.group(1)} x {dim_match.group(2)} x {dim_match.group(3)}"
            else:
                dimensions = 'N/A'
            
            # Extract total non-zeros
            nnz_match = re.search(r'Total non-zeros: (\d+)', section)
            total_nnz = nnz_match.group(1) if nnz_match else 'N/A'
            
            # Extract execution time from the summary
            exec_time_match = re.search(r'Execution time: ([\d.]+)ms', section)
            execution_time = exec_time_match.group(1) if exec_time_match else 'N/A'
            
            # Extract correctness
            correctness_match = re.search(r'Correctness: (PASS|FAIL|UNKNOWN)', section)
            correctness = correctness_match.group(1) if correctness_match else 'N/A'
            
            # Extract tile information
            tile_match = re.search(r'Tile: \d+ - time: ([\d.]+) ms nnz: (\d+) nFibers: (\d+) nSlc (\d+)', section)
            if tile_match:
                tile_time = tile_match.group(1)
                tile_nnz = tile_match.group(2)
                tile_nfibers = tile_match.group(3)
                tile_nslc = tile_match.group(4)
            else:
                tile_time = tile_nnz = tile_nfibers = tile_nslc = 'N/A'
            
            tensor_data.append({
                'dataset': dataset_name,
                'dimensions': dimensions,
                'total_nnz': total_nnz,
                'execution_time_ms': execution_time,
                'correctness': correctness,
                'tile_time_ms': tile_time,
                'tile_nnz': tile_nnz,
                'tile_nfibers': tile_nfibers,
                'tile_nslc': tile_nslc
            })
            
        except Exception as e:
            print(f"Error processing section for dataset: {e}")
            continue
    
    return tensor_data

def print_csv_data(tensor_data):
    """
    Print tensor data in CSV format.
    
    Args:
        tensor_data (list): List of dictionaries containing tensor data
    """
    if not tensor_data:
        print("No tensor data found.")
        return
    
    # CSV header
    headers = [
        'Dataset',
        'Dimensions',
        'Total_NNZ',
        'Execution_Time_ms',
        'Correctness',
        'Tile_Time_ms',
        'Tile_NNZ',
        'Tile_nFibers',
        'Tile_nSlc'
    ]
    
    print(','.join(headers))
    
    # Print data rows
    for data in tensor_data:
        row = [
            data['dataset'],
            data['dimensions'],
            data['total_nnz'],
            data['execution_time_ms'],
            data['correctness'],
            data['tile_time_ms'],
            data['tile_nnz'],
            data['tile_nfibers'],
            data['tile_nslc']
        ]
        print(','.join(str(item) for item in row))

def save_csv_data(tensor_data, output_file):
    """
    Save tensor data to CSV file.
    
    Args:
        tensor_data (list): List of dictionaries containing tensor data
        output_file (str): Path to output CSV file
    """
    if not tensor_data:
        print("No tensor data to save.")
        return
    
    headers = [
        'Dataset',
        'Dimensions',
        'Total_NNZ',
        'Execution_Time_ms',
        'Correctness',
        'Tile_Time_ms',
        'Tile_NNZ',
        'Tile_nFibers',
        'Tile_nSlc'
    ]
    
    try:
        with open(output_file, 'w') as f:
            # Write header
            f.write(','.join(headers) + '\n')
            
            # Write data rows
            for data in tensor_data:
                row = [
                    data['dataset'],
                    data['dimensions'],
                    data['total_nnz'],
                    data['execution_time_ms'],
                    data['correctness'],
                    data['tile_time_ms'],
                    data['tile_nnz'],
                    data['tile_nfibers'],
                    data['tile_nslc']
                ]
                f.write(','.join(str(item) for item in row) + '\n')
        
        print(f"\nCSV data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving CSV file: {e}")

def main():
    """Main function to run the tensor data extraction."""
    
    # Default log file path
    log_file_path = '/home/bhaskar/spTA/Week_11_TTMC_GPU/bcsf_all_datasets.log'
    
    # Check if custom path provided as command line argument
    if len(sys.argv) > 1:
        log_file_path = sys.argv[1]
    
    print(f"Extracting tensor data from: {log_file_path}")
    print("=" * 60)
    
    # Extract tensor data
    tensor_data = extract_tensor_data(log_file_path)
    
    if not tensor_data:
        print("No tensor data extracted. Please check the log file format.")
        return
    
    print(f"Found data for {len(tensor_data)} tensors\n")
    
    # Print CSV data to console
    print("CSV Output:")
    print("-" * 40)
    print_csv_data(tensor_data)
    
    # Save to CSV file
    output_file = os.path.join(os.path.dirname(log_file_path), 'tensor_data_extracted.csv')
    save_csv_data(tensor_data, output_file)
    
    print(f"\nSummary:")
    print(f"- Total datasets processed: {len(tensor_data)}")
    print(f"- Successful executions: {sum(1 for d in tensor_data if d['correctness'] == 'PASS')}")
    print(f"- Failed executions: {sum(1 for d in tensor_data if d['correctness'] in ['FAIL', 'ERROR'])}")
    print(f"- Unknown status: {sum(1 for d in tensor_data if d['correctness'] == 'UNKNOWN')}")

if __name__ == "__main__":
    main()
