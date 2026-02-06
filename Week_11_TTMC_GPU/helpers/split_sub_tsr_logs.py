import sys
from pathlib import Path


def split_log(log_path, out_dir):
    log_path = Path(log_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current_lines = []
    current_file_stem = None

    with log_path.open("r") as f:
        for line in f:
            if line.startswith("File:"):
                # Flush previous tensor block if any
                if current_file_stem is not None and current_lines:
                    out_path = out_dir / f"{current_file_stem}.log"
                    with out_path.open("w") as out_f:
                        out_f.writelines(current_lines)

                # Start new tensor block
                current_lines = [line]
                full_path = line.split("File:", 1)[1].strip()
                # e.g. /home/bhaskar/tensors_dataset/nell-2.tns -> nell-2
                stem = Path(full_path).stem
                current_file_stem = stem
            else:
                # Continue accumulating lines for current tensor
                if current_file_stem is not None:
                    current_lines.append(line)

    # Flush last tensor block
    if current_file_stem is not None and current_lines:
        out_path = out_dir / f"{current_file_stem}.log"
        with out_path.open("w") as out_f:
            out_f.writelines(current_lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python split_sub_tsr_logs.py <log_file> [output_dir]")
        sys.exit(1)

    log_file = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "logs"

    split_log(log_file, out_dir)


if __name__ == "__main__":
    main()

