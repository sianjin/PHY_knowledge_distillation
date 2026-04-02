function betaOpt = betaOptimization(results,mcs,cfgHE,beta)
resultIdx = true(1,size(results,2));
resultsUse = [results{resultIdx}];
sinrStore = cat(3,resultsUse.sinrStore);
perStore = cat(1,resultsUse.perStore);
mse = @(beta)awgnPerSnrFittingMse(sinrStore,perStore,'HE_SU',mcs,cfgHE.ChannelCoding,cfgHE.APEPLength,beta);
betaOpt = fminsearch(mse,beta);
end