#!/usr/bin/env bash
# profile_all_single_log.sh

LOGFILE="ncu_all_results.log"
echo "Profiling run started at $(date)" > "$LOGFILE"  # overwrite or create

METRICS="sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,sm__cycles_active.avg,sm__cycles_active.min,sm__cycles_active.max"
VERSIONS=( "./ttmc_v7.out" "./ttmc_v9.out" )  # list your executables here

for csf in *.csf; do
  for bin in "${VERSIONS[@]}"; do
    ver=$(basename "$bin" .out)
    echo "=== Dataset: $csf, Version: $ver ===" >> "$LOGFILE"
    ncu --set default \
        --metrics $METRICS \
        "$bin" "$csf" >> "$LOGFILE" 2>&1
    echo "" >> "$LOGFILE"
  done
done

echo "Profiling completed at $(date)" >> "$LOGFILE"
