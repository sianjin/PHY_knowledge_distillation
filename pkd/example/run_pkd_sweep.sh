#!/usr/bin/env bash
# run_pkd_sweep.sh  Run the PKD resource-usage benchmark across the same
# CBW x MIMO configuration sweep as run_eesm_sweep.sh, timing PKD at the
# SAME SNR grid each EESM-log-AR run discovered (read from that run's JSON
# output), one configuration per process.
#
# PKD is queried via --snr-list, which skips the training-dataset load
# entirely (PKD's continuous conditioning network needs no real data at
# those points, unlike EESM-log-AR's per-SNR calibration) -- this gives a
# peak-RSS number isolated from dataset-load overhead. Run run_eesm_sweep.sh
# FIRST so its JSON files (containing each configuration's real SNR grid)
# already exist in outDir.
#
# Usage:
#   ./run_pkd_sweep.sh [outDir] [numSequences] [sequenceLength] [modelPath]
#
# outDir must already contain the resource_usage_eesm_*.json files written
# by run_eesm_sweep.sh (same outDir passed to both scripts). numSequences
# and sequenceLength should match the EESM run for a directly comparable
# pair; this script's own PKD JSON output is written into the same outDir.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR/results}"
NUM_SEQ="${2:-10}"
SEQ_LEN="${3:-1000}"
MODEL_PATH="${4:-pkd/pkd_model.pt}"

CHANNEL_MODEL=2   # Model-B, matches run_eesm_sweep.sh
MCS=7

if [[ ! -d "$OUT_DIR" ]]; then
  echo "outDir '$OUT_DIR' does not exist -- run run_eesm_sweep.sh first." >&2
  exit 1
fi

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
  EESM_JSON="$OUT_DIR/resource_usage_eesm_${TAG}.json"

  if [[ ! -f "$EESM_JSON" ]]; then
    echo "Skipping ${TAG}: ${EESM_JSON} not found (run run_eesm_sweep.sh for this config first)." >&2
    continue
  fi

  # Pull the SNR grid out of the EESM JSON (avoids re-deriving or
  # hand-copying it, and guarantees PKD is timed at EXACTLY the same points).
  SNR_LIST="$(python -c "
import json
with open('$EESM_JSON') as f:
    d = json.load(f)
print(' '.join(str(s) for s in d['snr_values']))
")"

  echo "=== PKD: BW=${BW}, ${NT}x${NR}:${NSS}, MCS=${MCS}, N_seq=${NUM_SEQ}, SNR grid from ${EESM_JSON} ==="

  python -m pkd.example.evaluate_resource_usage \
    --channel-model "$CHANNEL_MODEL" --N-t "$NT" --N-r "$NR" --BW "$BW" --N-ss "$NSS" --MCS "$MCS" \
    --method pkd --snr-list $SNR_LIST \
    --num-sequences "$NUM_SEQ" --sequence-length "$SEQ_LEN" \
    --model-path "$MODEL_PATH" \
    --out-dir "$OUT_DIR"

  echo "=== Done: ${TAG} ==="
  echo
done

echo "All configurations complete. Results in: $OUT_DIR"
