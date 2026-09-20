#!/usr/bin/env bash
# run_eesm_sweep.sh  Run the EESM-log-AR resource-usage benchmark across the
# same CBW x MIMO configuration sweep as phy/runtime-benchmark's MATLAB
# sweep (runMeasureResourceSweep.sh), one configuration per process.
#
# EESM-log-AR must calibrate from real teacher sequences, so it always loads
# the training dataset to discover the configuration's actual waterfall SNR
# grid -- there is no "isolated, no-dataset-load" mode for it (unlike PKD's
# --snr-list path in evaluate_resource_usage.py). Each configuration is
# still run as its own OS process, matching the MATLAB sweep's isolation
# rationale (no state -- caches, torch/OMP thread pools, page cache -- shared
# between configurations).
#
# Each run also saves a JSON file (via --out-dir) recording the discovered
# SNR grid alongside the timing results. run_pkd_sweep.sh reads that JSON
# to time PKD at the SAME SNR points for a directly comparable pair, without
# re-deriving or hand-copying the grid.
#
# Usage:
#   ./run_eesm_sweep.sh [outDir] [numSequences] [sequenceLength]
#
# Defaults match the MATLAB sweep's clean N_seq=10 diagnostic convention;
# pass numSequences=50 for the paper's full Table III/IV numbers once the
# configuration is confirmed clean.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR/results}"
NUM_SEQ="${2:-10}"
SEQ_LEN="${3:-1000}"

CHANNEL_MODEL=2   # Model-B, matches phy/runtime-benchmark's CH="Model-B"
MCS=7

mkdir -p "$OUT_DIR"
# Resolve to an absolute path BEFORE cd-ing to REPO_ROOT below, so a
# relative outDir (e.g. "results", meant relative to the caller's cwd) is
# not silently written relative to REPO_ROOT instead.
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# (BW, N_t, N_r, N_ss) sweep -- matches runMeasureResourceSweep.sh's
# CBW20/CBW40 x {1x1:1, 3x2:1, 4x2:2}
CONFIGS=(
  "20|1|1|1"
  "20|3|2|1"
  "20|4|2|2"
  "40|1|1|1"
  "40|3|2|1"
  "40|4|2|2"
)

cd "$REPO_ROOT"

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r BW NT NR NSS <<< "$cfg"
  TAG="CBW${BW}_ch${CHANNEL_MODEL}_${NT}x${NR}_${NSS}SS"

  echo "=== EESM-log-AR: BW=${BW}, ${NT}x${NR}:${NSS}, MCS=${MCS}, N_seq=${NUM_SEQ} ==="

  python -m pkd.example.evaluate_resource_usage \
    --channel-model "$CHANNEL_MODEL" --N-t "$NT" --N-r "$NR" --BW "$BW" --N-ss "$NSS" --MCS "$MCS" \
    --method eesm --num-sequences "$NUM_SEQ" --sequence-length "$SEQ_LEN" \
    --out-dir "$OUT_DIR"

  echo "=== Done: ${TAG} ==="
  echo
done

echo "All configurations complete. Results in: $OUT_DIR"
echo "Next: ./run_pkd_sweep.sh \"$OUT_DIR\" to time PKD at the same SNR grids."
