function Z = localMapNormForward2D(X, windowSize, alpha, beta, k)
% localMapNormForward2D   Perform cross channel normalization
%   Z = localMapNormForward2D(X, windowSize, alpha, beta, k) computes the
%   channel normalized version Z of X using the parameters specified by
%   windowSize, alpha, beta and k.
%
%   Inputs:
%   X - The input feature channels for a set of images. A (H)x(W)x(C)x(N) 
%       array.
%   windowSize - The number of channels to use for the normalization of 
%       each element.
%   alpha - Multiplier for the normalization term.
%   beta - Exponent for the normalization term.
%   k - Offset for the normalization term.
%
%   Output:
%   Z - The output feature channels for the images. A (H)x(W)x(C)x(N)
%       array.

%   Copyright 2016 The MathWorks, Inc.

numChannels = size(X,3);
numExamples = size(X,4);

alpha = alpha/windowSize;
XSquared = X.^2;
normalizers = zeros(size(X), 'like', X);
for n = 1:numExamples
    for c = 1:numChannels
        [startChannel, stopChannel] = iGetStartAndStopChannels(numChannels, windowSize, c);
        normalizers(:,:,c,n) = sum(XSquared(:,:,startChannel:stopChannel,n), 3);
        normalizers(:,:,c,n) = (k + alpha*normalizers(:,:,c,n)).^beta;
    end
end
Z = X./normalizers;

end

function [startChannel, stopChannel] = iGetStartAndStopChannels(numChannels, windowSize, channelIndex)
lookBehind = floor((windowSize - 1)/2);
lookAhead = windowSize - lookBehind - 1;
calculatedStartMap = channelIndex - lookBehind;
startChannel = max(calculatedStartMap, 1);
calculatedStopMap = channelIndex + lookAhead;
stopChannel = min(calculatedStopMap, numChannels);
end