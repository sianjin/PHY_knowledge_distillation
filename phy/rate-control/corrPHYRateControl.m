function corrPHYRateControl(cbw, chan, numTxRx, numSs, N_real, T, outFile)
%corrPHYRateControl Fig. 15 teacher: N closed-loop rate-adaptation runs.
%
%   corrPHYRateControl(cbw, chan, numTxRx, numSs, N_real, T, outFile) runs
%   N_real independent realizations of the closed-loop rate-adaptation
%   experiment over ONE common time-varying SNR trajectory, using the
%   MATLAB PHY simulator as the effective-SINR source. Results are saved to
%   outFile for the Python-side comparison (pkd/example/evaluate_rate_control.py).
%
%   Each realization uses an independent channel-fading RNG seed but the
%   SAME SNR trajectory and the SAME rate-control policy. Teacher and PKD
%   are compared statistically (MCS-selection probability, achieved-
%   throughput distribution), not realization-by-realization.
%
%   Defaults:
%     cbw     = "CBW40"
%     chan    = "Model-B"
%     numTxRx = [3 2]
%     numSs   = 2
%     N_real  = 100
%     T       = 1000
%     outFile = fullfile(fileparts(mfilename('fullpath')), 'teacher_rate_control.mat')

if nargin < 1 || isempty(cbw),     cbw = "CBW40";    end
if nargin < 2 || isempty(chan),    chan = "Model-B"; end
if nargin < 3 || isempty(numTxRx), numTxRx = [3 2];  end
if nargin < 4 || isempty(numSs),   numSs = 2;        end
if nargin < 5 || isempty(N_real),  N_real = 100;     end
if nargin < 6 || isempty(T),       T = 1000;         end
if nargin < 7 || isempty(outFile)
    outFile = fullfile(fileparts(mfilename('fullpath')), 'teacher_rate_control.mat');
end

mcsList = 0:9;

% --- Fixed PHY configuration ---
cfgHE = wlanHESUConfig;
cfgHE.ChannelBandwidth = char(cbw);
cfgHE.APEPLength = 1000;         % payload length in bytes
cfgHE.ChannelCoding = 'LDPC';

% --- Common deterministic SNR trajectory (shared with the PKD student) ---
snrTraj = snrTrajectory(T);
trajFile = fullfile(fileparts(mfilename('fullpath')), 'snr_trajectory.mat');
save(trajFile, 'snrTraj', 'T', 'cbw', 'chan', 'numTxRx', 'numSs', 'mcsList');
fprintf('corrPHYRateControl: saved SNR trajectory to %s\n', trajFile);

% --- Calibrated EESM beta for every MCS on this slice (cached) ---
betaVec = betaTable(cbw, chan, numTxRx, numSs, mcsList);

% --- Build the fixed simParams (SNR index is irrelevant; overridden) ---
% getBox0SimParams needs a valid isnr for the (slice, MCS); use MCS 0, isnr 1.
simParams = getBox0SimParams(chan, numTxRx, numSs, 0, cfgHE, 1e3, T, 1);

% --- Preallocate ---
mcsAll     = zeros(N_real, T, 'uint8');
effSINRAll = zeros(N_real, T, 'single');
perInstAll = zeros(N_real, T, 'single');
errorAll   = zeros(N_real, T, 'uint8');
txPeriodAll = zeros(N_real, 1);

mcsInit = 4; % start mid-table

baseSeed = 54321;
seed = int64(baseSeed + (0:N_real-1).');

parfor n = 1:N_real
    rng(double(seed(n)), 'twister');
    r = box0RateControl(simParams, betaVec, mcsList, snrTraj, mcsInit);
    mcsAll(n, :)     = r.mcs(:).';
    effSINRAll(n, :) = single(r.effSINR(:).');
    perInstAll(n, :) = single(r.perInst(:).');
    errorAll(n, :)   = r.errorFlag(:).';
    txPeriodAll(n)   = r.txPeriod;
end

txPeriod = txPeriodAll(1);
payloadBits = cfgHE.APEPLength * 8;

% Achieved goodput uses actual PHY airtime. txPeriod is the interval between
% channel samples and includes idle time, so it is not a packet duration.
packetDurationByMCS = zeros(1, numel(mcsList));
cfgDuration = simParams.Config;
sampleRate = wlanSampleRate(cfgDuration);
for i = 1:numel(mcsList)
    cfgMCS = cfgDuration;
    cfgMCS.MCS = mcsList(i);
    psduLength = getPSDULength(cfgMCS);
    tx = wlanWaveformGenerator(zeros(psduLength * 8, 1, 'int8'), cfgMCS, ...
        'IdleTime', 0, 'WindowTransitionTime', 0);
    packetDurationByMCS(i) = size(tx, 1) / sampleRate;
end

successMask = (errorAll == 0);
successBits = double(sum(successMask, 2)) * payloadBits;
totalAirtimeSeconds = zeros(N_real, 1);
for i = 1:numel(mcsList)
    totalAirtimeSeconds = totalAirtimeSeconds + ...
        sum(mcsAll == mcsList(i), 2) * packetDurationByMCS(i);
end
throughputMbps = successBits ./ totalAirtimeSeconds / 1e6;

meta = struct('cbw', char(cbw), 'chan', char(chan), 'numTxRx', numTxRx, ...
    'numSs', numSs, 'N_real', N_real, 'T', T, 'payloadBits', payloadBits, ...
    'txPeriod', txPeriod, 'mcsInit', mcsInit, 'mcsList', mcsList, ...
    'betaVec', betaVec, 'packetDurationByMCS', packetDurationByMCS);

% This output is small enough for v7, which MATLAB can read and write over
% \\wsl.localhost and Python can load with scipy.io.loadmat.
tempFile = [tempname(fileparts(outFile)), '.mat'];
tempCleanup = onCleanup(@() deleteIfPresent(tempFile));
save(tempFile, 'mcsAll', 'effSINRAll', 'perInstAll', 'errorAll', ...
    'throughputMbps', 'snrTraj', 'txPeriod', 'payloadBits', 'seed', ...
    'packetDurationByMCS', 'totalAirtimeSeconds', 'meta', '-v7');
[moved, message] = movefile(tempFile, outFile, 'f');
if ~moved
    error('corrPHYRateControl:SaveFailed', ...
        'Unable to replace %s with the completed MAT file: %s', outFile, message);
end
clear tempCleanup

fprintf('corrPHYRateControl: saved %d runs x %d packets to %s\n', N_real, T, outFile);
fprintf('  mean achieved throughput: %.2f Mbps (std %.2f)\n', ...
    mean(throughputMbps), std(throughputMbps));
fprintf('  mean MCS: %.2f, mean PER: %.4f\n', ...
    mean(double(mcsAll(:))), mean(double(errorAll(:))));
end

function deleteIfPresent(path)
if isfile(path)
    delete(path);
end
end
