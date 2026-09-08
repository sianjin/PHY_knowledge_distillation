function tbl = calibrateSnrRange(cbw, chan, numTxRx, numSs, snrGrid, nRuns, T)
%calibrateSnrRange Find the SNR range that sweeps MCS ~1 to ~8 for a slice.
%
%   tbl = calibrateSnrRange(...) holds the average SNR CONSTANT at each
%   value of snrGrid, runs the closed-loop controller for a few short
%   realizations, and reports the settled MCS distribution. Use the result
%   to choose snrMin / snrMax for snrTrajectory.m so that the Fig. 15
%   trajectory genuinely sweeps the slice's usable MCS range (rather than
%   pinning at the MCS floor or ceiling).
%
%   Inputs (all optional)
%     cbw     = "CBW40"
%     chan    = "Model-B"
%     numTxRx = [3 2]
%     numSs   = 2
%     snrGrid = 0:5:45        constant SNR values to probe (dB)
%     nRuns   = 5             independent channel realizations per SNR
%     T       = 800           packets per run (last 60%% used as "settled";
%                             800 is enough for the controller to converge
%                             from the mid-table start at any SNR)
%
%   Output: table with columns
%     SNR, medianMCS, meanMCS, modeMCS, p10MCS, p90MCS, meanPER
%   over the settled portion of all runs at that SNR. THE TABLE IS THE
%   REFERENCE -- the printed "Suggested" line is only a starting point.
%
%   Suggested rule:
%     snrMin = lowest SNR where the typical MCS is off the floor AND the
%              link is workable: medianMCS >= 1 AND meanPER < PER_USABLE
%              (0.15). (The naive "medianMCS <= 1" rule matched the whole
%              outage plateau -- meanPER ~ 1 -- and kept recommending
%              snrMin = 0. A p90MCS >= 1 gate is too weak: at a barely-
%              usable SNR the controller still sits at MCS 0 most of the
%              time, so the trajectory trough becomes a flat MCS-0 stripe.)
%     snrMax = lowest SNR where medianMCS >= MCS_HIGH (8).

if nargin < 1 || isempty(cbw),     cbw = "CBW40";     end
if nargin < 2 || isempty(chan),    chan = "Model-B";  end
if nargin < 3 || isempty(numTxRx), numTxRx = [3 2];   end
if nargin < 4 || isempty(numSs),   numSs = 2;         end
if nargin < 5 || isempty(snrGrid), snrGrid = 0:5:45;  end
if nargin < 6 || isempty(nRuns),   nRuns = 5;         end
if nargin < 7 || isempty(T),       T = 800;           end

mcsList = 0:9;

cfgHE = wlanHESUConfig;
cfgHE.ChannelBandwidth = char(cbw);
cfgHE.APEPLength = 1000;
cfgHE.ChannelCoding = 'LDPC';

betaVec = betaTable(cbw, chan, numTxRx, numSs, mcsList);
simParams = getBox0SimParams(chan, numTxRx, numSs, 0, cfgHE, 1e3, T, 1);

settledFrom = floor(0.4 * T) + 1; % ignore the initial transient

SNR = snrGrid(:);
medianMCS = zeros(size(SNR));
meanMCS   = zeros(size(SNR));
modeMCS   = zeros(size(SNR));
p10MCS    = zeros(size(SNR));
p90MCS    = zeros(size(SNR));
meanPER   = zeros(size(SNR));

for j = 1:numel(snrGrid)
    snrConst = snrGrid(j) * ones(T, 1);
    settledMcs = [];
    settledErr = [];
    for n = 1:nRuns
        rng(70000 + 100*j + n, 'twister');
        r = box0RateControl(simParams, betaVec, mcsList, snrConst, 4);
        settledMcs = [settledMcs; double(r.mcs(settledFrom:end))];       %#ok<AGROW>
        settledErr = [settledErr; double(r.errorFlag(settledFrom:end))]; %#ok<AGROW>
    end
    medianMCS(j) = median(settledMcs);
    meanMCS(j)   = mean(settledMcs);
    modeMCS(j)   = mode(settledMcs);
    p10MCS(j)    = prctile(settledMcs, 10);
    p90MCS(j)    = prctile(settledMcs, 90);
    meanPER(j)   = mean(settledErr);
    fprintf('SNR %5.1f dB -> median MCS %d (mean %.2f, p10 %d, p90 %d), PER %.3f\n', ...
        snrGrid(j), medianMCS(j), meanMCS(j), p10MCS(j), p90MCS(j), meanPER(j));
end

tbl = table(SNR, medianMCS, meanMCS, modeMCS, p10MCS, p90MCS, meanPER);

% --- Suggested snrMin / snrMax (starting point only; read the table) -----
PER_USABLE = 0.15;   % below this the link is workable, not in outage
MCS_HIGH   = 8;      % "top of the sweep" target

usable = (medianMCS >= 1) & (meanPER < PER_USABLE);
lo = SNR(find(usable, 1, 'first'));
hi = SNR(find(medianMCS >= MCS_HIGH, 1, 'first'));
if ~isempty(lo) && ~isempty(hi) && hi > lo
    fprintf(['\nSuggested (verify against the table): ' ...
             'snrMin = %g, snrMax = %g\n'], lo, hi);
    fprintf('  snrTrajectory(1000, %g, %g)\n', lo, hi);
else
    if isempty(lo)
        fprintf('\nNo SNR on the grid is "usable" (medianMCS >= 1 & meanPER < %.2f); widen snrGrid upward.\n', PER_USABLE);
    end
    if isempty(hi)
        fprintf('\nNo SNR on the grid reaches median MCS %d; widen snrGrid upward.\n', MCS_HIGH);
    end
end
end
