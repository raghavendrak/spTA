import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log(log_path):
    datasets = []
    current = None
    in_block_section = False

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                in_block_section = False
                continue

            if line.startswith("File:"):
                # Start a new dataset
                if current is not None:
                    datasets.append(current)
                current = {
                    "file": line.split("File:", 1)[1].strip(),
                    "order": None,
                    "dims": [],
                    "blocks_per_dim": [],
                    "block_idx": [],
                    "nnz": [],
                }
                in_block_section = False
            elif line.startswith("Order:"):
                if current is not None:
                    current["order"] = int(line.split("Order:", 1)[1].strip())
            elif line.startswith("Dims:"):
                if current is not None:
                    parts = line.split("Dims:", 1)[1].strip().split()
                    current["dims"] = [int(x) for x in parts]
            elif line.startswith("Blocks per dim:"):
                if current is not None:
                    parts = line.split("Blocks per dim:", 1)[1].strip().split()
                    current["blocks_per_dim"] = [int(x) for x in parts]
            elif line.startswith("BlockIndex"):
                in_block_section = True
            elif in_block_section and current is not None:
                # lines: "<block_idx> <nnz>"
                parts = line.split()
                if len(parts) >= 2:
                    b = int(parts[0])
                    c = int(parts[1])
                    current["block_idx"].append(b)
                    current["nnz"].append(c)

    if current is not None:
        datasets.append(current)

    return datasets


def plot_datasets(datasets, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        if not ds["block_idx"]:
            continue

        fname = Path(ds["file"]).stem
        title = f"{fname} (order={ds['order']}, dims={ds['dims']})"

        plt.figure(figsize=(10, 4))
        plt.bar(ds["block_idx"], ds["nnz"], width=1.0)
        plt.xlabel("Sub-tensor block index")
        plt.ylabel("Number of non-zeros")
        plt.title(title)
        plt.tight_layout()

        out_path = out_dir / f"{fname}_sub_tsr_nnz.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot: {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_sub_tsr_nz_cnt.py <log_file> [output_dir]")
        sys.exit(1)

    log_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "plots"

    datasets = parse_log(log_path)
    if not datasets:
        print("No datasets found in log.")
        sys.exit(1)

    plot_datasets(datasets, out_dir)


if __name__ == "__main__":
    main()

