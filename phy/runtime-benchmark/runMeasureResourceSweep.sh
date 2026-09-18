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
#   ./runMeasureResourceSweep.sh [outDir] [logDir] [N_seq]
#
# N_seq (default 50, matching the paper's Table III/IV numbers) is the
# number of sequences per SNR point. At N_seq=50 the full 6-configuration
# sweep is roughly a day and a half of sequential wall-clock time (each
# configuration's own N_seq=50/numSnr=10 cost, summed) -- to get a much
# faster read on whether the isolation fix produced low per-SNR variance
# and the expected CBW20-vs-CBW40 trend before committing to that, pass a
# smaller N_seq, e.g.:
#   ./runMeasureResourceSweep.sh results logs 10
# Diagnostic-N_seq output files are tagged _Nseq<N> by measureResource.m so
# they never collide with the full N_seq=50 result; once the trend looks
# right, re-run the configurations you need (or all of them) at the
# default N_seq=50 for the numbers that actually go in the paper.
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
N_SEQ="${3:-50}"
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

  echo "=== Running ${CBW}, ${CH}, ${NUMTXRX}, numSs=${NUMSS}, MCS=${MCS}, N_seq=${N_SEQ} ==="
  echo "    (fresh matlab -batch process; log: ${LOG_FILE})"

  matlab -batch "addpath(\"${SCRIPT_DIR}\"); measureResource(\"${CBW}\",\"${CH}\",${NUMTXRX},${NUMSS},${MCS},\"${OUT_DIR}\",${N_SEQ})" \
    2>&1 | tee "$LOG_FILE"

  echo "=== Done: ${TAG} ==="
  echo
done

echo "All configurations complete. Results in: $OUT_DIR"
echo "Check each log in $LOG_DIR for a 'HighVariance' warning before trusting a result."
