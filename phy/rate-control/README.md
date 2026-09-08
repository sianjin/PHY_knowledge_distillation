# rate-control/ — Closed-loop rate adaptation under a time-varying configuration (Fig. 15)

This workflow addresses Reviewer 1's comment that the framework is motivated by
time-varying configurations, yet all other experiments fix `C_t = C`. Here the
average SNR follows a common deterministic trajectory and the MCS is selected
packet-by-packet by a rate controller, so `C_t != C`.

It is a copy of `runtime-benchmark/` with a rate-control driver added. The
shared PHY helpers (`box0Validation.m`, `calculateSINR.m`, `getBox0*Params.m`,
`spatialCorrelation.m`, `tgaxLinkPerformanceModel.m`, `mcs2beta.m`,
`betaOptimization.m`, `awgnPerSnrFittingMse.m`, ...) are unchanged.
`main.m`, `corrPHY.m`, `corrPHYSim.m` are the original runtime-benchmark
entry points and are **not** used here.

## What Fig. 15 tests

Whether replacing the PHY simulator with the PKD student preserves *closed-loop
rate-adaptation behaviour* under a time-varying configuration — **statistically**,
not by reproducing the same stochastic realization. Teacher and student share:

- the same time-varying SNR trajectory `{SNR_t}` (`snrTrajectory.m`), and
- the same rate-control policy (`rateController.m`).

The only difference is the source of effective SINR:

| System  | Effective-SINR source                         |
|---------|-----------------------------------------------|
| Teacher | MATLAB PHY simulator (this folder)            |
| Student | PKD log-AR process (`pkd/`, Python)           |

Fidelity hierarchy: effective-SINR → PER → **closed-loop rate-adaptation**.

## Fixed static slice

Model-B, `3x2:2`, CBW40 (same slice as the Fig. 14 PER–SNR waterfall).
MCS 0–9 adapted dynamically. Payload 1000 bytes, LDPC.

## Files

| File | Role |
|---|---|
| `snrTrajectory.m` | Deterministic common `{SNR_t}`: **one slow sinusoid cycle** (`f = 1`, starts high, dips to `snrMin` near the midpoint, rises back) + fixed-seed AR(1) jitter. One slow cycle keeps the SNR quasi-static over ~100-packet windows so the controller settles and the Fig. 15(b) histograms stay tight. Saved to `snr_trajectory.mat`. |
| `rateController.m` | Shared EWMA-PER dual-threshold controller (`ALPHA=0.2`, `PER_LOW=0.02`, `PER_HIGH=0.1`, `UP_COUNT=3`). **Byte-for-byte twin of `pkd/rate_control.py`.** |
| `calibrateSnrRange.m` | Holds SNR constant on a grid, reports where the controller settles (median MCS per SNR), and suggests `snrMin`/`snrMax` so the trajectory sweeps ~MCS 1 to ~MCS 8. Run once per slice. |
| `betaTable.m` | Calibrates EESM `beta` for MCS 0–9 on the slice via `corrPHYVal`. Cached to `beta_table.mat` (keyed by slice; mismatched cache errors out). Outside the closed-loop path. |
| `box0RateControl.m` | One closed-loop realization: one TGax channel realization, per-packet effective SINR from EESM at the time-varying `N0_t`, coin flip vs. AWGN-LUT PER, `rateController` picks `MCS_{t+1}`. |
| `corrPHYRateControl.m` | `N_real` independent realizations (`parfor`, independent channel seeds, common trajectory). Computes whole-run and windowed achieved goodput. Saves `teacher_rate_control.mat`. |
| `main_rate_control.m` | Top-level driver. |

## How to run

```matlab
cd phy/rate-control

% 0. One-time per slice: calibrate the SNR range
tbl = calibrateSnrRange("CBW40", "Model-B", [3 2], 2);
disp(tbl)                       % read the "Suggested: snrMin=.. snrMax=.." line
%   -> put those into main_rate_control.m (snrMin / snrMax)

% 1. Smoke test (fast)
corrPHYRateControl("CBW40", "Model-B", [3 2], 2, 4, 200, [], 100, 20, 6, 40);

% 2. Full run
main_rate_control               % N_real = 100, T = 1000
```

First run calibrates `beta` for all 10 MCS (slow, cached afterwards in
`beta_table.mat`). Delete `beta_table.mat` to force recalibration.

## Outputs (consumed by `pkd/example/evaluate_rate_control.py`)

- **`snr_trajectory.mat`** — `snrTraj` (T×1), plus slice metadata.
- **`beta_table.mat`** — `betaVec` (1×10), `mcsList`, slice metadata.
- **`teacher_rate_control.mat`** (MATLAB v7; load in Python with
  `scipy.io.loadmat(..., simplify_cells=True)`):
  - `mcsAll` (N×T, uint8) — MCS used per packet
  - `effSINRAll` (N×T) — EESM effective SINR (dB)
  - `perInstAll` (N×T) — AWGN-LUT PER at (effSINR, MCS)
  - `errorAll` (N×T, uint8) — sampled packet-error flags
  - `throughputMbps` (N×1) — whole-run achieved goodput per run (airtime-
    weighted; tight spread, kept for the inset mean ± std, not plotted)
  - `segGoodputMbps` (N×nSeg), `goodputSamplesMbps` (N·nSeg × 1) —
    **time-resolved** achieved goodput over non-overlapping `segLen`-packet
    windows; `goodputSamplesMbps` is what Fig. 15(c) plots the CDF of
  - `segSnrMean` (1×nSeg) — mean SNR of each window
  - `packetDurationByMCS` (1×10) — HE waveform airtime per MCS
  - `snrTraj`, `txPeriod`, `payloadBits`, `seed`, `meta` (`meta` has
    `segLen`, `burnIn`, `segStarts`, `betaVec`, …; `txPeriod` is the
    channel-sample spacing, **not** packet airtime)

## Validation checklist (before the Python side)

- [ ] `calibrateSnrRange` brackets MCS 1–8; `snrMin`/`snrMax` in
      `main_rate_control.m` updated from its suggestion.
- [ ] `snrTrajectory(1000, snrMin, snrMax)` is one smooth slow cycle:
      starts near `snrMax`, dips to `snrMin` around packet 500, rises back.
- [ ] Smoke run completes; **at a fixed packet index the 100 runs cluster
      within ~1–2 adjacent MCS** (tight Fig. 15(b) band — this is the key
      check that motivated `f=1` / `UP_COUNT=3`).
- [ ] `mean MCS_t` tracks the SNR trajectory (high MCS near the peaks,
      MCS 0–1 at the trough), correlation > 0.9.
- [ ] `effSINRAll` correlates with `snrTraj` (broadcast across runs).
- [ ] Mean sampled PER between `PER_LOW=0.02` and `PER_HIGH=0.1`
      (typically ~0.03–0.08).
- [ ] `goodputSamplesMbps` has real spread (trough windows well below peak
      windows) — the Fig. 15(c) CDF should not be a vertical line.
- [ ] `throughputMbps` is a plausible spread (tens of Mbps for this slice).
