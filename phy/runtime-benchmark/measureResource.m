function results_summary = measureResource(CBW, CH, numTxRx, numSs, mcs, outDir, N_seq)
% measureResource  Single-process resource-usage measurement for the
% traditional PHY abstraction, AVERAGED OVER ALL SNR OPERATING POINTS
% (Table III/IV baseline). Reports CPU utilization and peak memory
% alongside wall-clock time; see pkd/example/evaluate_resource_usage.py
% for the matching PKD/EESM-log-AR measurement on the Python side, run
% over the SAME SNR grid so the three are directly comparable.
%
% Matches measureRuntimeParfor.m's convention of looping over numSnr = 10 SNR points
% (isnr = 1:10) for the same (CBW, CH, numTxRx, numSs, mcs), and reports
% the per-SNR wall-clock plus the average -- the same quantity
% measureRuntimeParfor.m reports as tAvg. Unlike that script, this function
% does not use parfor: it runs
% all 10 SNR points sequentially in ONE MATLAB process, because reporting
% the aggregate CPU%/memory of a 10-worker parfor pool would conflate "how
% many SNR points were batched in parallel" with "how much does one
% configuration cost" -- measureRuntimeParfor.m's tAvg already backs out per-unit time via
% *numCores, and CPU utilization/memory must be sampled over the same
% single-process, sequential-SNR-loop unit to be comparable against PKD's
% own single-process measurement.
%
% This is a FUNCTION (not a plain script) so that runMeasureResourceSweep.sh
% can launch it in its own isolated `matlab -batch` process per
% configuration via:
%   matlab -batch "measureResource(\"CBW40\",\"Model-B\",[3 2],2,7,'results')"
% Running each configuration in its own OS process (rather than editing
% variables and re-running inside one persistent MATLAB session) avoids any
% state carried over between configurations -- a lingering parallel pool,
% JIT/cache warm-up effects, or another process's leftover CPU load -- all
% of which otherwise show up as elevated per-SNR timing variance.
%
% N_seq (7th argument, optional, default 50) overrides the number of
% sequences per SNR point. The paper's Table III/IV numbers use N_seq=50;
% a smaller N_seq (e.g. 10) is useful for a quick multi-configuration
% variance/trend check across a full sweep without committing to the full
% multi-day run for every configuration up front. Output files are tagged
% with _Nseq<N> whenever N_seq != 50, so a diagnostic run never collides
% with the full N_seq=50 result.
%
% Requires MATLAB on Windows for the `memory` function (MemUsedMATLAB is
% Windows-only; see https://www.mathworks.com/help/matlab/ref/memory.html).
% CPU utilization uses `cputime`, which is cross-platform, but this script
% is intended to be run on the same Windows machine as the rest of
% phy/runtime-benchmark for a self-consistent Table III/IV entry.

if nargin < 6 || isempty(outDir)
    outDir = fileparts(mfilename('fullpath'));
end
if nargin < 7 || isempty(N_seq)
    N_seq = 50;           % sequences per SNR point (matches corrPHYSim.m / Table II/III)
end

numSnr  = 10;             % matches measureRuntimeParfor.m; loops isnr = 1:numSnr
T       = 1000;           % packets per sequence (matches corrPHYSim.m)
maxNumErrors  = 1e3;
maxNumPackets = T;

% ---- Fixed PHY configuration ----
cfgHE = wlanHESUConfig;
cfgHE.ChannelBandwidth = CBW;
cfgHE.APEPLength = 1000;      % payload length in bytes
cfgHE.ChannelCoding = 'LDPC';

fprintf('Configuration: %s, %s, %dx%d:%d, MCS %d, %d SNR points\n', ...
    CBW, CH, numTxRx(1), numTxRx(2), numSs, mcs, numSnr);

% ---- Offline (untimed) beta calibration -- excluded from the timed/
% measured region, matching corrPHYSim.m's convention. beta is calibrated
% once for this configuration (not per SNR point), same as measureRuntimeParfor.m/
% corrPHYSim.m. Uses corrPHYValSequential (a plain `for` loop) instead of
% corrPHYVal (parfor), so that no parallel pool is ever spawned during a
% resource-usage run -- whether a pool actually starts, and how long
% start-up/teardown takes, is not deterministic across machines or
% configurations, and that variability was itself showing up as
% inconsistent measurement conditions between configurations. ----
fprintf('Calibrating EESM beta (offline, untimed, sequential)...\n');
betaOpt = corrPHYValSequential(CBW, CH, mcs, numTxRx, numSs);
fprintf('  beta = %.4f\n', betaOpt);

% ---- Build one simParams struct per SNR point (untimed) ----
simParamsAll = cell(1, numSnr);
snrValues = zeros(1, numSnr);
for isnr = 1:numSnr
    simParamsAll{isnr} = getBox0SimParams(CH, numTxRx, numSs, mcs, cfgHE, maxNumErrors, maxNumPackets, isnr);
    snrValues(isnr) = simParamsAll{isnr}.SNR;
end
fprintf('  SNR grid (dB): %s\n', mat2str(snrValues));

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
    warning('measureResource:NoMemoryFcn', ...
        ['This is not Windows: MATLAB''s memory() function is unavailable, ' ...
         'so peak-memory will NOT be measured (only wall-clock and CPU%%). ' ...
         'Run this script on the Windows MATLAB machine for the full result.']);
    peakMemBytes = NaN;
end

cpuStart = cputime;
wallSweepStart = tic;

% ---- Timed region: single process, sequential SNR loop, no parfor ----
wallTimePerSnr = zeros(1, numSnr);
for isnr = 1:numSnr
    simParams = simParamsAll{isnr};

    snrWallStart = tic;
    gamma_eff        = zeros(N_seq, T, 'single'); %#ok<NASGU>
    packet_abs_error = zeros(N_seq, T, 'uint8');  %#ok<NASGU>
    for n = 1:N_seq
        rng(double(seed(n)), 'twister');

        results = box0Simulation(simParams, betaOpt);

        gamma_eff(n, :)        = single(results.snreffStore(1:T)); %#ok<NASGU>
        packet_abs_error(n, :) = uint8(results.perAbsStore(1:T));  %#ok<NASGU>

        if haveMemoryFcn
            m = memory;
            peakMemBytes = max(peakMemBytes, m.MemUsedMATLAB);
        end
    end
    wallTimePerSnr(isnr) = toc(snrWallStart);
    fprintf('  SNR=%6.2f dB (isnr=%d/%d): %.4f s\n', snrValues(isnr), isnr, numSnr, wallTimePerSnr(isnr));
end

wallTimeTotal = toc(wallSweepStart);
cpuTime = cputime - cpuStart;
cpuUtilPct = 100 * cpuTime / wallTimeTotal;

totalPacketsPerSnr = N_seq * T;
wallTimeAvg = mean(wallTimePerSnr);
wallTimeStd = std(wallTimePerSnr);
wallTimeRelStdPct = 100 * wallTimeStd / wallTimeAvg;
wallTimePerSeqAvg = wallTimeAvg / N_seq;
wallTimePerPacketMsAvg = wallTimeAvg / totalPacketsPerSnr * 1e3;
throughputPacketsPerSecAvg = totalPacketsPerSnr / wallTimeAvg;

% ---- Report ----
fprintf('\n');
fprintf('=========================================================\n');
fprintf('Resource-Usage Result: Traditional PHY Abstraction (MATLAB)\n');
fprintf('=========================================================\n');
fprintf('SNR points:            %d  (%.1f to %.1f dB)\n', numSnr, min(snrValues), max(snrValues));
fprintf('Sequences x packets:   %d x %d per SNR point\n', N_seq, T);
fprintf('Average per-SNR time:  %.4f s  (std %.4f s, %.1f%% relative)  [%.4f s/sequence, %.4f ms/packet]\n', ...
    wallTimeAvg, wallTimeStd, wallTimeRelStdPct, wallTimePerSeqAvg, wallTimePerPacketMsAvg);
if wallTimeRelStdPct > 10
    warning('measureResource:HighVariance', ...
        ['Per-SNR wall-clock relative std is %.1f%% (>10%%). This usually means ' ...
         'the process was not isolated (background load, another MATLAB/worker ' ...
         'process running concurrently, thermal throttling, etc.) -- treat this ' ...
         'result with caution and prefer re-running on an otherwise-idle machine.'], ...
        wallTimeRelStdPct);
end
fprintf('Total wall-clock:      %.4f s  (sum over all SNR points, single process)\n', wallTimeTotal);
fprintf('Throughput:            %.1f packets/s (per-SNR average)\n', throughputPacketsPerSecAvg);
fprintf('CPU time (cputime):    %.4f s\n', cpuTime);
fprintf('CPU utilization:       %.1f%%  (100%% = 1 core; %d logical cores available)\n', ...
    cpuUtilPct, feature('numcores'));
if haveMemoryFcn
    fprintf('Peak memory (MATLAB):  %.1f MB  (whole SNR sweep)\n', peakMemBytes / 1e6);
else
    fprintf('Peak memory (MATLAB):  NOT MEASURED (memory() requires Windows)\n');
end
fprintf('=========================================================\n');
fprintf('\nCompare directly against pkd/example/evaluate_resource_usage.py\n');
fprintf('(same num-sequences/sequence-length, same SNR grid via --snr-list\n');
fprintf('or by matching the dataset''s own grid) for the corresponding PKD\n');
fprintf('and EESM-log-AR (Python) numbers.\n');

results_summary = struct( ...
    'CBW', CBW, 'CH', CH, 'numTxRx', numTxRx, 'numSs', numSs, 'mcs', mcs, ...
    'snrValues', snrValues, 'N_seq', N_seq, 'T', T, ...
    'wallTimePerSnr', wallTimePerSnr, 'wallTimeAvg', wallTimeAvg, 'wallTimeStd', wallTimeStd, ...
    'wallTimeRelStdPct', wallTimeRelStdPct, ...
    'wallTimeTotal', wallTimeTotal, 'cpuTime', cpuTime, 'cpuUtilPct', cpuUtilPct, ...
    'peakMemBytes', peakMemBytes, 'throughputPacketsPerSecAvg', throughputPacketsPerSecAvg);

if ~exist(outDir, 'dir')
    mkdir(outDir);
end
if N_seq == 50
    outName = sprintf('resource_usage_%s_%s_%dx%d_%dSS.mat', ...
        CBW, CH, numTxRx(1), numTxRx(2), numSs);
else
    % Tag the filename when N_seq deviates from the paper's N_seq=50, so a
    % reduced-N_seq diagnostic run (e.g. a quick variance/isolation check)
    % never overwrites or gets confused with the full N_seq=50 result.
    outName = sprintf('resource_usage_%s_%s_%dx%d_%dSS_Nseq%d.mat', ...
        CBW, CH, numTxRx(1), numTxRx(2), numSs, N_seq);
end
outPath = fullfile(outDir, outName);
save(outPath, 'results_summary');
fprintf('\nSaved results_summary to %s\n', outPath);

end
