function [dX,dBeta,dGamma] = batchNormalizationBackward(dZ, X, gamma, epsilon, batchMean, batchInvVar) %#ok<INUSL>
% Back-propagation using batch normalization layer on the host
% NB: batchInvVar is actually 1./sqrt(var(X) + epsilon)

%   Copyright 2016-2017 The MathWorks, Inc.

X0 = (X - batchMean);
Xnorm = X0 .*  batchInvVar;
m = numel(X) ./ size(X,3); % total number of elements in batch per activation

% First get the gradient of the function w.r.t. input (x)
% See Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network
% Training by Reducing Internal Covariate Shift" for details.
dl_dXnorm = dZ .* gamma;
dl_dvar = -0.5 .* iSumAllExcept3D(dl_dXnorm .* X0) .*  batchInvVar.^3;
% Ignore the final term from the paper since it sum(X0) should be zero if
% batchMean is correct.
dl_dmean = -iSumAllExcept3D(dl_dXnorm .*  batchInvVar);% - 2.*dl_dvar.*iSumAllExcept3D(X0)./m;

dX = (dl_dXnorm .* batchInvVar) ...
    + (2 .* dl_dvar .* X0 ./ m) ...
    + (dl_dmean ./ m);
    

% Now w.r.t to the parameters beta and gamma.
dBeta = iSumAllExcept3D(dZ);
dGamma = iSumAllExcept3D(dZ .* Xnorm);
end


function out = iSumAllExcept3D(in)
% Helper to sum a 4D array in all dimensions except the third:
%  (HxWxCxN) -> (1x1xCx1)
in = permute(in,[1 2 4 3]);
in = reshape(in, [], 1, size(in,4));
% Now have (H*W*N)x1xC, so can sum in first dimension
out = sum(in,1);

end