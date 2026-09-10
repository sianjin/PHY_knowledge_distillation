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
| `rateController.m` | Shared EWMA-PER dual-threshold controller: raise MCS when EWMA PER < `PER_LOW=0.03`, lower when > `PER_HIGH=0.10`, hold in the dead-band. `ALPHA=0.1`, `UP_COUNT=1` (ascent tracks the sweep), `DOWN_COUNT=2` (a single unlucky packet error no longer forces a step down — keeps the Fig. 15(b) histograms tight). **Byte-for-byte twin of `pkd/rate_control.py`.** |
| `calibrateSnrRange.m` | Holds SNR constant on a grid, reports where the controller settles (median MCS per SNR). **The printed table is the reference**; the one-line "Suggested" is a starting point (`snrMin` = lowest SNR with `medianMCS >= 1` and `meanPER < 0.15`; `snrMax` = lowest SNR with `medianMCS >= 8`). Run once per slice. |
| `betaTable.m` | Calibrates EESM `beta` for MCS 0–9 on the slice via `corrPHYVal`. Cached to `beta_table.mat` (keyed by slice; mismatched cache errors out). Outside the closed-loop path. |
| `box0RateControl.m` | One closed-loop realization: one TGax channel realization, per-packet effective SINR from EESM at the time-varying `N0_t`, coin flip vs. AWGN-LUT PER, `rateController` picks `MCS_{t+1}`. |
| `corrPHYRateControl.m` | `N_real` independent realizations (`parfor`, independent channel seeds, common trajectory). Computes whole-run and windowed achieved goodput. Saves `teacher_rate_control.mat`. |
| `main_rate_control.m` | Top-level driver. |
| `genRateControllerFixture.m` | Runs `rateController.m` over a few crafted PER sequences and saves the MCS traces to `rate_controller_fixture.mat` for the Python parity test (`pkd/tests/test_rate_control_parity.py`). Re-run whenever `rateController.m` changes. |

## How to run

```matlab
cd phy/rate-control

% 0. One-time per slice: calibrate the SNR range (parfor over SNR points)
tbl = calibrateSnrRange("CBW40", "Model-B", [3 2], 2);
%   read the printed table; set snrMin/snrMax in main_rate_control.m

% 1. Smoke test (fast: 4 runs, full T so the sweep + convergence show)
corrPHYRateControl("CBW40", "Model-B", [3 2], 2, 4, 1000, [], 200, 50, 25, 45);

% 2. Full run
main_rate_control               % N_real = 100, T = 1000

% 3. (once, or after editing rateController.m) fixture for the Python parity test
genRateControllerFixture
```

First run calibrates `beta` for all 10 MCS (slow, cached afterwards in
`beta_table.mat`). Delete `beta_table.mat` to force recalibration.

Then build Fig. 15 on the Python side:

```bash
python -m pkd.example.evaluate_rate_control      # reads the two .mat files above
```

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

- [ ] `calibrateSnrRange` brackets MCS ~1–9; `snrMin`/`snrMax` in
      `main_rate_control.m` set (currently 25 / 45 for this slice — 25 is
      the lowest SNR with medianMCS >= 1, so the trough is not a flat
      MCS-0 stripe).
- [ ] `snrTrajectory(1000, 25, 45)` is one smooth slow cycle: starts near
      45 dB, dips to 25 dB around packet 500, rises back.
- [ ] `mean MCS_t` tracks the SNR trajectory (high MCS near the peaks,
      low MCS at the trough), correlation > 0.9. The ascending half
      (packets ~500–1000) should reach roughly the same MCS as the
      descending half at equal SNR — no large hysteresis gap.
- [ ] `effSINRAll` correlates with `snrTraj` (broadcast across runs).
- [ ] Overall sampled PER near the operating band (~0.05–0.10). If it is
      ~0.2, the controller is failing to climb — revisit `UP_COUNT` /
      `ALPHA`.
- [ ] The Fig. 15(b) MCS-selection band is diffuse (~2–3 MCS wide on the
      ramps) because the threshold controller dithers; this is expected
      and is fine as long as it **matches** between teacher and PKD. The
      controller is a fixed shared test harness, not a proposed optimal
      rate-control scheme, so it is deliberately left simple.
- [ ] `goodputSamplesMbps` has real spread (trough windows well below peak
      windows) — the Fig. 15(c) CDF should not be a vertical line.
- [ ] `throughputMbps` is a plausible spread (tens of Mbps for this slice).
