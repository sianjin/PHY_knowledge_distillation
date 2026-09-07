function H = helperPerfectChannelEstimate(pathGains,pathFilters,ofdmInfo,varargin)
%helperPerfectChannelEstimate perfect channel estimation
%   H = helperPerfectChannelEstimate(PATHGAINS,PATHFILTERS,OFDMINFO)
%   performs perfect channel estimation.
%
%   H is an array of size Nst-by-Nsym-by-Nt-by-Nr-by-Nl. Nst is the number
%   of active subcarriers. Nsym is the number of OFDM symbols. Nt is the
%   number of transmit antennas. Nr is the number of receive antennas. Nl
%   is the number of links.
%
%   PATHGAINS must be an array of size Ns-by-Np-by-Nt-by-Nr-by-Nl, where Ns
%   is the number of path gain samples and, Np is the number of paths. The
%   channel impulse response is averaged across all samples and summed
%   across all transmit antennas and receive antennas before timing
%   estimation.
%
%   PATHFILTERS must be a matrix of size Np-by-Nh where Nh is the number of
%   impulse response samples. The path filters is assumed to be the same
%   for all links.
%
%   OFDMINFO is a structure with the these fields:
%     FFTLength        - FFT length
%     CPLength         - Cyclic prefix length
%     ActiveFFTIndices - Indices of active subcarriers within the FFT in
%                        the range [1, NFFT]
%
%   H = helperPerfectChannelEstimate(...,OFFSET) performs perfect channel
%   estimation given a timing offset, OFFSET. If not provided the ideal
%   offset is calculated internally.
%
%   OFFSET is a vector of length Nl indicating estimated timing offset, an
%   integer number of samples relative to the first sample of the channel
%   impulse response reconstructed from PATHGAINS and PATHFILTERS.

%   See also channelDelay.

%   Copyright 2019-2026 The MathWorks, Inc.

H = wlan.internal.phy.l2sm.perfectChannelEstimate(pathGains,pathFilters,ofdmInfo,varargin{:});

end