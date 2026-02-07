%% mcs 4
clear all
tStart = tic;% Simulation Parameters
mcs = 0:9; % Vector of MCS to simulate between 0 and 9
numTxRx = [4 2]; % Matrix of MIMO schemes, each row is [numTx numRx]
numSs = 1; % Number of spatial streams
chan = "Model-B"; % String array of delay profiles to simulate
% maxnumberrors = 100e3;  % The maximum number of packet errors at an SNR point 
% maxNumPackets = 100e3; % The maximum number of packets at an SNR point
maxnumberrors = 1e2;  % The maximum number of packet errors at an SNR point
maxNumPackets = 1e3; % The maximum number of packets at an SNR point

% Fixed PHY configuration for all simulations
cfgHE = wlanHESUConfig;
cfgHE.ChannelBandwidth = 'CBW40'; % Channel bandwidth
bandwidth = cfgHE.ChannelBandwidth;
cfgHE.APEPLength = 1000;          % Payload length in bytes
cfgHE.ChannelCoding = 'LDPC';     % Channel coding

% Generate a structure array of simulation configurations. Each element is
% one SNR point to simulate.
simParams = getBox0SimParams(chan,numTxRx,numSs,mcs,cfgHE,maxnumberrors,maxNumPackets);
snrs = [simParams.SNR];

% Simulate each configuration
results = cell(1,numel(simParams));
parfor isim = 1:numel(simParams)  % Use 'parfor' to speed up the simulation
    results{isim} = box0Simulation(simParams(isim));
end
tEnd = toc(tStart);
fname_I = sprintf('snrPer_%s_%s_%s-by-%s-by-%s_MCS%s.mat',bandwidth,char(chan),num2str(numTxRx(1)),num2str(numTxRx(2)),num2str(numSs),num2str(mcs));
save(fname_I,'results','mcs','numTxRx','numSs','chan','cfgHE','maxNumPackets','snrs','tEnd')

plotPERvsSNR(simParams,results)
