#!/usr/bin/env python3
import re
import csv

LOGFILE = "ncu_all_results.log"

# Map metric names to readable labels
metrics = {
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved",
    "sm__maximum_warps_per_active_cycle_pct": "theoretical",
    "sm__cycles_active.avg": "avg",
    "sm__cycles_active.min": "min",
    "sm__cycles_active.max": "max",
}

metric_re = re.compile(r'^\s*(\S+)\s+\S+\s+([\d,\.]+)')

results = {}  # results[dataset][version][metric] = value

# Parse the log file
with open(LOGFILE) as f:
    current_ds = current_ver = None
    for line in f:
        header = re.match(r'^=== Dataset: (.*), Version: (.*) ===', line)
        if header:
            current_ds, current_ver = header.group(1), header.group(2)
            results.setdefault(current_ds, {}).setdefault(current_ver, {})
            continue

        m = metric_re.match(line)
        if m and m.group(1) in metrics and current_ds and current_ver:
            label = metrics[m.group(1)]
            value = float(m.group(2).replace(',', ''))  # remove commas
            # print(m.group(2))
            # print(f"{current_ds} {current_ver} {label} {value}")
            results[current_ds][current_ver][label] = float(value)

# Write to CSV
with open("occupancy_summary.csv", "w", newline='') as csvf:
    writer = csv.writer(csvf)

    versions = sorted({ver for ds in results.values() for ver in ds})
    header = ["Dataset"]
    for ver in versions:
        header += [f"{ver}-theoretical", f"{ver}-achieved", f"{ver}-imbalance%"]
    writer.writerow(header)

    for ds in sorted(results):
        row = [ds]
        for ver in versions:
            entry = results[ds].get(ver, {})
            theoretical = entry.get("theoretical", "")
            achieved = entry.get("achieved", "")
            imbalance = ""
            if all(k in entry for k in ["avg", "min", "max"]) and entry["avg"] > 0:
                imbalance = (entry["max"] - entry["min"]) / entry["avg"] * 100
                # print(f"({entry['max']} - {entry['min']}) / {entry['avg']} * 100")
                # print(f"Imbalance for {ds} {ver}: {imbalance}")
                imbalance = round(imbalance, 4)
            row += [theoretical, achieved, imbalance]
        writer.writerow(row)

print("Parsed occupancy_summary.csv with load imbalance")
