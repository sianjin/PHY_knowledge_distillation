% main_measure_resource  Single-configuration, single-process resource-usage
% measurement for the traditional PHY abstraction (Table III/IV baseline).
%
% Addresses reviewer comments:
%   1. "The evaluation reports only execution time. The authors do not
%      measure CPU/GPU utilization and memory footprint."
%   2. "The reported runtime reduction ... is evaluated only on a CPU."
%      (This script IS the CPU-only baseline being characterized; see
%      pkd/example/evaluate_resource_usage.py for the matching PKD/
%      EESM-log-AR measurement on the Python side, run with the SAME
%      single-process, single-configuration methodology so the two are
%      comparable.)
%
% Unlike main.m/corrPHYSim.m (which use parfor across 10 SNR points to
% produce the wall-clock number in Table III via tAvg = tEnd/numSnr*
% numCores), this script deliberately runs ONE SNR point in ONE MATLAB
% process, no parfor. Reporting the aggregate CPU%/memory of a 10-worker
% parfor pool would conflate "how many configurations were batched" with
% "how much does one configuration cost" -- the parfor wall-clock number
% already backs out per-unit time via *numCores, and CPU utilization/
% memory must be reported on the same per-configuration, single-worker
% basis to be a fair unit for comparison against PKD's own single-process
% measurement.
%
% Requires MATLAB on Windows for the `memory` function (MemUsedMATLAB is
% Windows-only; see https://www.mathworks.com/help/matlab/ref/memory.html).
% CPU utilization uses `cputime`, which is cross-platform, but this script
% is intended to be run on the same Windows machine as the rest of
% phy/runtime-benchmark for a self-consistent Table III/IV entry.

clear; clc;

% ---- Fixed configuration (edit to match the desired Table III/IV row) ----
CBW     = "CBW40";
CH      = "Model-B";
numTxRx = [3 2];
numSs   = 2;
mcs     = 7;
isnr    = 6;            % index into the per-(config,MCS) SNR grid, 1..10
                         % (isnr=6 -> SNR=41 dB for Model-B 3x2:2 MCS7,
                         % matching the SNR used in the PKD/EESM-log-AR
                         % resource-usage comparison: pkd/example/
                         % evaluate_resource_usage.py --snr 41)
N_seq   = 50;            % sequences (matches corrPHYSim.m / Table II/III)
T       = 1000;          % packets per sequence (matches corrPHYSim.m)
maxNumErrors  = 1e3;
maxNumPackets = T;

% ---- Fixed PHY configuration ----
cfgHE = wlanHESUConfig;
cfgHE.ChannelBandwidth = CBW;
cfgHE.APEPLength = 1000;      % payload length in bytes
cfgHE.ChannelCoding = 'LDPC';

fprintf('Configuration: %s, %s, %dx%d:%d, MCS %d, SNR index %d\n', ...
    CBW, CH, numTxRx(1), numTxRx(2), numSs, mcs, isnr);

% ---- Offline (untimed) beta calibration -- excluded from the timed/
% measured region, matching corrPHYSim.m's convention ----
fprintf('Calibrating EESM beta (offline, untimed)...\n');
betaOpt = corrPHYVal(CBW, CH, mcs, numTxRx, numSs);
fprintf('  beta = %.4f\n', betaOpt);

% ---- Build the single-SNR-point simulation parameters ----
simParams = getBox0SimParams(CH, numTxRx, numSs, mcs, cfgHE, maxNumErrors, maxNumPackets, isnr);
fprintf('  SNR = %g dB\n', simParams.SNR);

% ---- Preallocate ----
gamma_eff        = zeros(N_seq, T, 'single');
packet_abs_error = zeros(N_seq, T, 'uint8');

baseSeed = 12345;
seed = int64(baseSeed + (0:N_seq-1)');

% ---- Resource-usage instrumentation ----
% MemUsedMATLAB: bytes currently used by MATLAB (Windows only).
% cputime: total CPU seconds consumed by this MATLAB process since it
% started (cross-platform); comparing its delta to wall-clock over the
% same region gives CPU utilization, capturing multi-threaded BLAS/JIT
% activity within a single MATLAB process (>100% is possible and expected
% since MATLAB's linear algebra is internally multi-threaded even without
% parfor).
haveMemoryFcn = ispc;   % `memory` is documented Windows-only
if haveMemoryFcn
    memStart = memory;
    peakMemBytes = memStart.MemUsedMATLAB;
else
    warning('main_measure_resource:NoMemoryFcn', ...
        ['This is not Windows: MATLAB''s memory() function is unavailable, ' ...
         'so peak-memory will NOT be measured (only wall-clock and CPU%%). ' ...
         'Run this script on the Windows MATLAB machine for the full result.']);
    peakMemBytes = NaN;
end

cpuStart = cputime;
wallStart = tic;

% ---- Timed region: single-process, single-configuration, no parfor ----
for n = 1:N_seq
    rng(double(seed(n)), 'twister');

    results = box0Simulation(simParams, betaOpt);

    gamma_eff(n, :)        = single(results.snreffStore(1:T));
    packet_abs_error(n, :) = uint8(results.perAbsStore(1:T));

    if haveMemoryFcn
        m = memory;
        peakMemBytes = max(peakMemBytes, m.MemUsedMATLAB);
    end
end

wallTime = toc(wallStart);
cpuTime  = cputime - cpuStart;
cpuUtilPct = 100 * cpuTime / wallTime;

totalPackets = N_seq * T;
wallTimePerSeq = wallTime / N_seq;
wallTimePerPacketMs = wallTime / totalPackets * 1e3;
throughputPacketsPerSec = totalPackets / wallTime;

% ---- Report ----
fprintf('\n');
fprintf('=========================================================\n');
fprintf('Resource-Usage Result: Traditional PHY Abstraction (MATLAB)\n');
fprintf('=========================================================\n');
fprintf('Sequences x packets:   %d x %d (%d total packets)\n', N_seq, T, totalPackets);
fprintf('Wall-clock time:       %.4f s  (%.4f s/sequence, %.4f ms/packet)\n', ...
    wallTime, wallTimePerSeq, wallTimePerPacketMs);
fprintf('Throughput:            %.1f packets/s\n', throughputPacketsPerSec);
fprintf('CPU time (cputime):    %.4f s\n', cpuTime);
fprintf('CPU utilization:       %.1f%%  (100%% = 1 core; %d logical cores available)\n', ...
    cpuUtilPct, feature('numcores'));
if haveMemoryFcn
    fprintf('Peak memory (MATLAB):  %.1f MB\n', peakMemBytes / 1e6);
else
    fprintf('Peak memory (MATLAB):  NOT MEASURED (memory() requires Windows)\n');
end
fprintf('=========================================================\n');
fprintf('\nCompare directly against pkd/example/evaluate_resource_usage.py\n');
fprintf('run with --method pkd / --method eesm, same num-sequences/sequence-length,\n');
fprintf('for the matching single-process, single-configuration PKD and\n');
fprintf('EESM-log-AR (Python) numbers.\n');

results_summary = struct( ...
    'CBW', CBW, 'CH', CH, 'numTxRx', numTxRx, 'numSs', numSs, 'mcs', mcs, ...
    'snr', simParams.SNR, 'N_seq', N_seq, 'T', T, ...
    'wallTime', wallTime, 'cpuTime', cpuTime, 'cpuUtilPct', cpuUtilPct, ...
    'peakMemBytes', peakMemBytes, 'throughputPacketsPerSec', throughputPacketsPerSec);
save(fullfile(fileparts(mfilename('fullpath')), 'resource_usage_result.mat'), 'results_summary');
fprintf('\nSaved results_summary to resource_usage_result.mat\n');
