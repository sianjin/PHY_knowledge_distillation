% main_rate_control  Top-level driver for the Fig. 15 closed-loop
% rate-adaptation experiment (MATLAB PHY teacher).
%
% Produces:
%   snr_trajectory.mat       - common deterministic time-varying SNR input
%   beta_table.mat           - calibrated EESM beta for MCS 0-9 on the slice
%   teacher_rate_control.mat - N closed-loop realizations (per-packet MCS,
%                              effective SINR, PER, errors; whole-run and
%                              windowed achieved goodput)
%
% The Python PKD student reads snr_trajectory.mat and teacher_rate_control.mat
% via pkd/example/evaluate_rate_control.py to build Fig. 15.
%
% Fixed static slice: Model-B, 3x2:2, CBW40 (same slice as the Fig. 14
% PER-SNR waterfall). Only MCS is time-varying, so C_t ~= C.

clear; clc;

cbw     = "CBW40";
chan    = "Model-B";
numTxRx = [3 2];
numSs   = 2;

% ---- Step 0 (one-time): calibrate the SNR range for this slice ----------
% Hold SNR constant on a grid, see where the controller settles, and pick
% snrMin/snrMax so the trajectory sweeps roughly MCS 1 to MCS 8.
%   tbl = calibrateSnrRange(cbw, chan, numTxRx, numSs);
%   disp(tbl)
% Then set snrMin/snrMax below. Note the "Suggested:" line uses median MCS
% <= 1 for snrMin, which can pick an outage SNR (PER ~ 1, MCS pinned at 0);
% prefer the lowest SNR where the link is actually usable (meanPER < ~0.1
% and MCS starts moving off 0). For the Model-B 3x2:2 CBW40 slice the
% calibration table gives: SNR 15 -> meanPER 0.08, MCS leaving 0;
% SNR 45 -> median MCS 8.
snrMin = 15;
snrMax = 45;

% ---- Smoke test first (fast) -------------------------------------------
% corrPHYRateControl(cbw, chan, numTxRx, numSs, 4, 1000, [], 200, 50, snrMin, snrMax);

% ---- Full run ---------------------------------------------------------
N_real = 100;
T      = 1000;
segLen = 200;   % windowed-goodput window for Fig. 15(c)
burnIn = 50;    % drop initial controller transient
corrPHYRateControl(cbw, chan, numTxRx, numSs, N_real, T, [], segLen, burnIn, snrMin, snrMax);
