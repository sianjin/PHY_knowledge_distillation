#!/usr/bin/env bash
# runMeasureResourceSweep.sh  Run the resource-usage sweep (CBW x MIMO config)
# as a set of fully isolated MATLAB processes, one per configuration.
#
# Each configuration is launched as its own `matlab -batch` invocation, i.e.
# a fresh OS process with no state (parallel pools, JIT/cache warm-up,
# variables) carried over from the previous configuration. This is
# deliberately more expensive than editing measureResource.m's arguments and
# re-running inside one persistent MATLAB session, but it removes
# cross-configuration contention as a confound when comparing runtime across
# configurations -- run each configuration on an otherwise-idle machine and
# do not run anything else in parallel with this script.
#
# Usage:
#   ./runMeasureResourceSweep.sh [outDir] [logDir]
#
# Results are saved as one .mat file per configuration in outDir (default:
# results/), named resource_usage_<CBW>_<CH>_<Nt>x<Nr>_<Nss>SS.mat by
# measureResource.m itself. Console output for each run is tee'd to logDir
# (default: logs/) for later inspection (e.g. to check the reported
# wallTimeRelStdPct warning).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR/results}"
LOG_DIR="${2:-$SCRIPT_DIR/logs}"
CH="Model-B"
MCS=7

mkdir -p "$OUT_DIR" "$LOG_DIR"

# (CBW, numTxRx, numSs) sweep -- matches the configurations already measured:
# CBW20/CBW40 x {1x1:1, 3x2:1, 4x2:2}
CONFIGS=(
  "CBW20|[1 1]|1"
  "CBW20|[3 2]|1"
  "CBW20|[4 2]|2"
  "CBW40|[1 1]|1"
  "CBW40|[3 2]|1"
  "CBW40|[4 2]|2"
)

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r CBW NUMTXRX NUMSS <<< "$cfg"
  TAG="${CBW}_${CH}_$(echo "$NUMTXRX" | tr -d '[] ' | tr ' ' 'x')_${NUMSS}SS"
  LOG_FILE="$LOG_DIR/measureResource_${TAG}.log"

  echo "=== Running ${CBW}, ${CH}, ${NUMTXRX}, numSs=${NUMSS}, MCS=${MCS} ==="
  echo "    (fresh matlab -batch process; log: ${LOG_FILE})"

  matlab -batch "addpath(\"${SCRIPT_DIR}\"); measureResource(\"${CBW}\",\"${CH}\",${NUMTXRX},${NUMSS},${MCS},\"${OUT_DIR}\")" \
    2>&1 | tee "$LOG_FILE"

  echo "=== Done: ${TAG} ==="
  echo
done

echo "All configurations complete. Results in: $OUT_DIR"
echo "Check each log in $LOG_DIR for a 'HighVariance' warning before trusting a result."
