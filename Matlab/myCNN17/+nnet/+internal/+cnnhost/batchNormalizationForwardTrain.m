function [Z,batchMean,batchInvVar] = batchNormalizationForwardTrain(X, beta, gamma, epsilon)
% Forward batch normalization on the host, training phase
% Returns the layer output and the batch mean and inverse variance
    
%   Copyright 2016-2017 The MathWorks, Inc.

% We want (H*W*N)-by-1-by-C array
Xc = permute(X,[1 2 4 3]);
Xc = reshape(Xc, [], 1, size(Xc,4));

batchMean = mean(Xc,1);
batchInvVar = 1./sqrt(var(Xc,1,1) + epsilon);

scale = gamma .* batchInvVar;
offset = beta - batchMean.*scale;

Z = scale.*X + offset;

end
