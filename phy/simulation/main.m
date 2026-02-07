% Main function to run simulation
% Loop over multiple bandwidth, MCS and SNR points

% Set parallel pool to use a different storage location
cluster = parcluster('Processes');
cluster.JobStorageLocation = tempdir;  % or another location with space
saveProfile(cluster);
delete(gcp('nocreate'));  % Close any existing pool

CBWstring = ["CBW20","CBW40"];
chanString = ["Model-B","Model-D"];
for cbw = 1:2
    for chan = 1:2
        for mcs = 0:9
            for isnr = 1:10
                corrPHY(CBWstring(cbw),chanString(chan),mcs,isnr);
            end
        end
    end
end