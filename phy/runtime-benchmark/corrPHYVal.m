function betaOpt = corrPHYVal(cbw,chan,mcs,numTxRx,numSs)
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

% Simulate each configuration using non-optimized beta
results = cell(1,numel(simParams));
beta = mcs2beta(mcs);
parfor isim = 1:numel(simParams)  % Use 'parfor' to speed up the simulation
    results{isim} = box0Validation(simParams(isim),beta);
end
betaOpt = betaOptimization(results,mcs,cfgHE,beta);
end
