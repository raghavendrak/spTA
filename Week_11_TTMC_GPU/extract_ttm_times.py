#!/usr/bin/env python3
"""
Python script to extract CUDA TTM kernel times from log output and format results.
"""

import re
import sys
from tabulate import tabulate

def extract_ttm_times(log_content):
    """
    Extract all CUDA TTM Kernel times from log content and return total time.
    """
    # Pattern to match CUDA TTM Kernel times
    ttm_pattern = r'\[CUDA\s+TTM\s+Kernel\]:\s+([0-9]+\.[0-9]+)\s+s\s+spent'
    
    matches = re.findall(ttm_pattern, log_content)
    total_time = sum(float(time) for time in matches)
    
    return total_time, matches

def parse_results_file(results_file):
    """
    Parse the results file and return formatted data.
    """
    results = []
    
    try:
        with open(results_file, 'r') as f:
            lines = f.readlines()
        
        # Skip header line
        for line in lines[1:]:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    dataset = parts[0]
                    time_str = parts[1]
                    
                    if time_str == 'ERROR':
                        results.append([dataset, 'ERROR'])
                    else:
                        try:
                            time_val = float(time_str)
                            time_val /= 10 #the kernel is runnning 10 times
                            results.append([dataset, f"{time_val:.6f}"])
                        except ValueError:
                            results.append([dataset, 'ERROR'])
        
        return results
    
    except FileNotFoundError:
        print(f"Results file not found: {results_file}")
        return []

def main():
    if len(sys.argv) > 1:
        # If log content is provided as argument, extract times
        log_content = sys.argv[1]
        total_time, individual_times = extract_ttm_times(log_content)
        print(f"Individual CUDA TTM times: {individual_times}")
        print(f"Total CUDA TTM time: {total_time:.6f}s")
    else:
        # Parse and display results file
        results_file = "/tmp/ttm_results.txt"
        results = parse_results_file(results_file)
        
        if results:
            # Sort results by dataset name
            results.sort(key=lambda x: x[0])
            
            headers = ["Dataset", "Total CUDA TTM Kernel Time (s)"]
            print(tabulate(results, headers=headers, tablefmt='grid'))
            
            # Calculate statistics for successful runs
            successful_times = []
            for row in results:
                if row[1] != 'ERROR':
                    try:
                        successful_times.append(float(row[1]))
                    except ValueError:
                        pass
            
            if successful_times:
                print(f"\nStatistics:")
                print(f"  Successful runs: {len(successful_times)}")
                print(f"  Failed runs: {len(results) - len(successful_times)}")
                print(f"  Average time: {sum(successful_times) / len(successful_times):.6f}s")
                print(f"  Min time: {min(successful_times):.6f}s")
                print(f"  Max time: {max(successful_times):.6f}s")
        else:
            print("No results found or results file is empty.")

if __name__ == "__main__":
    main()
