#add tensor order and dimensions to the beginning of the file after reading COO file

def analyze_and_prepend_header(input_path, output_path):
    import os

    max_indices = []
    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip():  # skip empty lines
                continue
            parts = line.strip().split()
            indices = list(map(int, parts[:-1]))  # all but last are indices
            if not max_indices:
                max_indices = [0] * len(indices)
            for i, idx in enumerate(indices):
                if idx > max_indices[i]:
                    max_indices[i] = idx

    order = len(max_indices)
    dimensions = [str(d) for d in max_indices]  # assuming 1-based indexing

    # Now write to output file
    with open(output_path, 'w') as out:
        out.write(f"{order}\n")
        out.write(" ".join(dimensions) + "\n")
        with open(input_path, 'r') as f:
            for line in f:
                out.write(line)
    
    # Atomically replace the original file with the new file
    os.replace(output_path, input_path)

import sys

if len(sys.argv) < 2:
    print("Usage: python preprocesor.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
analyze_and_prepend_header(input_file, "../../tensors_dataset/temp.tns")
