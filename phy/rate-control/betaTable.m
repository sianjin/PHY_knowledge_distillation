function betaVec = betaTable(cbw, chan, numTxRx, numSs, mcsList, cacheFile)
%betaTable Calibrated EESM beta for every MCS on the fixed Fig. 15 slice.
%
%   betaVec = betaTable(cbw, chan, numTxRx, numSs, mcsList, cacheFile)
%   returns a vector of calibrated EESM beta values, one per entry of
%   mcsList, for the fixed static configuration used by the closed-loop
%   rate-control experiment (Fig. 15). This mirrors how the existing
%   workflows calibrate beta with corrPHYVal/betaOptimization; calibration
%   is done ONCE up front and is outside the closed-loop path.
%
%   If cacheFile exists it is loaded and returned. Otherwise beta is
%   calibrated for each MCS and saved to cacheFile.
%
%   Defaults:
%     cbw      = "CBW40"
%     chan     = "Model-B"
%     numTxRx  = [3 2]
%     numSs    = 2
%     mcsList  = 0:9
%     cacheFile= fullfile(fileparts(mfilename('fullpath')), 'beta_table.mat')

if nargin < 1 || isempty(cbw),     cbw = "CBW40";     end
if nargin < 2 || isempty(chan),    chan = "Model-B";  end
if nargin < 3 || isempty(numTxRx), numTxRx = [3 2];   end
if nargin < 4 || isempty(numSs),   numSs = 2;         end
if nargin < 5 || isempty(mcsList), mcsList = 0:9;     end
if nargin < 6 || isempty(cacheFile)
    cacheFile = fullfile(fileparts(mfilename('fullpath')), 'beta_table.mat');
end

if exist(cacheFile, 'file')
    S = load(cacheFile, 'betaVec', 'mcsList');
    if isequal(S.mcsList(:).', mcsList(:).')
        betaVec = S.betaVec;
        fprintf('betaTable: loaded cached beta from %s\n', cacheFile);
        return;
    end
end

betaVec = zeros(1, numel(mcsList));
for i = 1:numel(mcsList)
    mcs = mcsList(i);
    fprintf('betaTable: calibrating beta for MCS %d ...\n', mcs);
    betaVec(i) = corrPHYVal(char(cbw), char(chan), mcs, numTxRx, numSs);
    fprintf('betaTable: MCS %d -> beta = %.4f\n', mcs, betaVec(i));
end

save(cacheFile, 'betaVec', 'mcsList', 'cbw', 'chan', 'numTxRx', 'numSs');
fprintf('betaTable: saved calibrated beta to %s\n', cacheFile);
end
