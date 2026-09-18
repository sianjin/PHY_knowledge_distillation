function betaOpt = corrPHYValSequential(cbw,chan,mcs,numTxRx,numSs)
% corrPHYValSequential  Same beta calibration as corrPHYVal, but with a
% plain `for` loop instead of `parfor`. Used by measureResource.m so that
% no parallel pool is spawned at any point during a resource-usage run --
% whether or not a pool actually starts (and how long start-up/teardown
% takes) is not deterministic across machines/configurations, and even
% though calibration itself is untimed, spinning a pool up and down right
% before the timed region can still perturb machine state at that
% boundary. Keep this in sync with corrPHYVal.m (used by main.m, where the
% parfor speedup is wanted) aside from the loop type.

maxNumErrors = 1e3;  % The maximum number of packet errors at an SNR point
maxNumPackets = 1e3; % The maximum number of packets at an SNR point

% Fixed PHY configuration for all simulations
cfgHE = wlanHESUConfig;
cfgHE.ChannelBandwidth = cbw; % Channel bandwidth
cfgHE.APEPLength = 1000;          % Payload length in bytes
cfgHE.ChannelCoding = 'LDPC';     % Channel coding

% Generate a structure array of simulation configurations. Each element is
% one SNR point to simulate.
simParams = getBox0ValParams(chan,numTxRx,numSs,mcs,cfgHE,maxNumErrors,maxNumPackets);

% Simulate each configuration sequentially (no parfor, no parallel pool)
results = cell(1,numel(simParams));
beta = mcs2beta(mcs);
for isim = 1:numel(simParams)
    results{isim} = box0Validation(simParams(isim),beta);
end
betaOpt = betaOptimization(results,mcs,cfgHE,beta);
end
