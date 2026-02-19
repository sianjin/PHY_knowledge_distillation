% Main function to run simulation
% Loop over multiple bandwidth, MCS and SNR points

% Set parallel pool to use a different storage location
cluster = parcluster('Processes');
cluster.JobStorageLocation = tempdir;  % or another location with space
saveProfile(cluster);
delete(gcp('nocreate'));  % Close any existing pool

CBWstring = ["CBW20","CBW40"];
chanString = ["Model-B","Model-D"];
numTxRx = [3 1];
numSs = 1;
for cbw = 1:2
    for chan = 1:2
        for mcs = 0:9
            betaOpt = corrPHYVal(CBWstring(cbw),chanString(chan),mcs,numTxRx,numSs);
            for isnr = 1:10
                corrPHYSim(CBWstring(cbw),chanString(chan),mcs,isnr,numTxRx,numSs,betaOpt);
            end
        end
    end
end