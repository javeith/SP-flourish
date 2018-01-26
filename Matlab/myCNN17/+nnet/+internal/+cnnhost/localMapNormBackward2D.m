function dLossdX = localMapNormBackward2D(Z, dLossdZ, X, windowSize, alpha, beta, k) %#ok<INUSL>
% localMapNormBackward2D   Perform backpropagation for local map normalization
%
% Inputs:
% Z - The output from the local map normalization layer. Not used in the
% current implementation but present to match cuDNN.
% dLossdZ - The derivative of the loss function with respect to the output
% of the normalization layer. A (H)x(W)x(C)x(N) array.
% X - The input feature maps for a set of images. A (H)x(W)x(C)x(N) array.
% windowSize - The number of maps to use for the normalization of each
% element.
% alpha - Multiplier for the normalization term.
% beta - Exponent for the normalization term.
% k - Offset for the normalization term.
%
% Output:
% dLossdZ - The derivative of the loss function with respect to the input
% of the normalization layer. A (H)x(W)x(C)x(N) array.

%   Copyright 2015-2016 The MathWorks, Inc.

numChannels = size(X,3);
numExamples = size(X,4);
alpha = alpha/windowSize;

XSquared = X.^2;
dZdX = zeros(size(X),'like',X);
for n = 1:numExamples
    for c = 1:numChannels
        [startMap, stopMap] = iGetStartAndStopMaps(numChannels, windowSize, c);
        localSumSq = sum(XSquared(:,:,startMap:stopMap,n), 3);
        y = (k + alpha*localSumSq);
        dZdX(:,:,c,n) = y.^-beta - ((2*alpha*beta*localSumSq).*y.^(-beta - 1));
    end
end
dLossdX = dLossdZ .* dZdX;

end

function [startMap, stopMap] = iGetStartAndStopMaps(numMaps, windowSize, mapIndex)
lookBehind = floor((windowSize - 1)/2);
lookAhead = windowSize - lookBehind - 1;
calculatedStartMap = mapIndex - lookBehind;
startMap = max(calculatedStartMap, 1);
calculatedStopMap = mapIndex + lookAhead;
stopMap = min(calculatedStopMap, numMaps);
end
