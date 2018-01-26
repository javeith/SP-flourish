function dLossdW = convolveBackwardFilter2D(X, W, dLossdZ, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    strideHeight, strideWidth)
% convolveBackwardFilter2D   Backpropagate through a convolutional layer to
% get the derivative with respect to the filters.
%
% Inputs:
% X - The input to the convolutional layer. An (H)x(W)x(C)x(N) array.
% W - The filters for the convolutional layer. We only pass these so that
% we can get their dimensions. An (R)x(S)x(C)x(K) array.
% dLossdZ - The derivative of the loss with respect to the output of the
% convolutional layer. An (H-R+1)x(W-S+1)x(K)x(N) array.
% padTop - Padding on the top.
% padLeft - Padding on the left.
% padBottom - Padding on the bottom.
% padRight - Padding on the right.
% strideHeight - The stride in the y direction.
% strideWidth - The stride in the x direction.
%
% Output:
% dLossdW - The derivative of the loss with respect to the filters. An
% (R)x(S)x(C)x(K) array.

%   Copyright 2016-2017 The MathWorks, Inc.

% The height and width of the filters. Note that this cannot be deduced
% from dLossdZ and X.

filterHeight = size(W,1);
filterWidth = size(W,2);

numExamples = size(dLossdZ,4);
numFilters = size(W,4);

outputHeight = size(dLossdZ,1);
outputWidth = size(dLossdZ,2);

X = permute(X, [1 2 4 3]);
dLossdZ = permute(dLossdZ, [1 2 4 3]);

% Upsample the input if the stride is greater than 1.
if (strideHeight > 1) || (strideWidth > 1)
    upsamplingKernel = zeros(strideHeight, strideWidth);
    upsamplingKernel(1,1) = 1;
    dLossdZUpsampled = zeros((outputHeight-1)*strideHeight+1, (outputWidth-1)*strideWidth+1, 'like', dLossdZ);
    for n = 1: numFilters
        for k = 1: numExamples
            dLossdZUpsampledOneChannel = kron(dLossdZ(:,:,k,n), upsamplingKernel);
            dLossdZUpsampled(:,:,k,n) = dLossdZUpsampledOneChannel(1:end-strideHeight+1,1:end-strideWidth+1);
        end
    end
    dLossdZ = dLossdZUpsampled;
end

% Compute the convolution
dLossdW = nnet.internal.cnnhost.convolveBackward(X, dLossdZ, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    strideHeight, strideWidth);

dLossdW = dLossdW(1:filterHeight, 1:filterWidth,:,:);
dLossdW = permute(dLossdW, [1 2 4 3]);
end



