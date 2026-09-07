% Main function to run simulation
% Loop over multiple bandwidth, MCS and SNR points

CBWstring = ["CBW20","CBW40"];
chanString = ["Model-B","Model-D"];
numTxRx = [3 2];
numSs = 1;
numSnr = 10;
numCores = 10; % number of parallel cores
cbw = 1;
chan = 1;
mcs = 7;
CBW = CBWstring(cbw);
CH = chanString(chan);
betaOpt = corrPHYVal(CBWstring(cbw),chanString(chan),mcs,numTxRx,numSs);

% Average runtime calculation
tStart = tic; 
parfor isnr = 1:numSnr
    corrPHYSim(CBW,CH,mcs,isnr,numTxRx,numSs,betaOpt);
end
tEnd = toc(tStart);
tAvg = tEnd/numSnr*numCores;
fprintf('Average runtime for %s, %s, %dx%d: %d, MCS %d, is %.4f\n', ...
    CBW, CH, numTxRx(1), numTxRx(2), numSs, mcs, tAvg);
