function dLossdX = poolingAverageBackward2D(Z, dLossdZ, X, ...
    poolHeight, poolWidth, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    strideHeight, strideWidth, ...
    includePadding) %#ok<INUSL>
% poolingAverageBackward2D   Perform backpropagation for mean pooling
%
% Inputs:
% Z - The output from the pooling layer. This is not used here, but is
% dLossdZ - The derivative of the loss function with respect to the output
% of the pooling layer. A
% floor((H + padTop + padBottom - poolHeight)/strideHeight + 1) x
% floor((W + padLeft + padRight - poolWidth)/strideWidth + 1) x
% (C) x (N) array.
% X - The input to the pooling layer. A (H)x(W)x(C)x(N) array.
% added for consistency with the cuDNN interface.
% poolHeight - The height of a pooling region
% poolWidth - The width of a pooling region
% padTop - Padding on the top.
% padLeft - Padding on the left.
% padBottom - Padding on the bottom.
% padRight - Padding on the right.
% strideHeight - The stride in the y direction.
% strideWidth - The stride in the x direction.
% includePadding (optional) - Specifies if padding should be included in
% the average. The default is true.
%
% Output:
% dLossdX - The derivative of the loss function with respect to the input
% of the pooling layer. A (H)x(W)x(C)x(N) array.


%   Copyright 2015-2017 The MathWorks, Inc.

imageHeight = size(X,1);
imageWidth = size(X,2);
numMaps = size(X,3);
numExamples = size(X,4);

% Work out if we want to include padding.
if nargin < 12
    includePadding = true;
end

% Allocate memory for the output.
dLossdX = zeros(size(X), 'like', X);

for n = 1:numExamples
    for c = 1:numMaps
        % Perform backward mean pooling by first upsampling the input
        % matrix, and then convolving it with a filter of ones.
        if((strideHeight > 1) || (strideWidth > 1))
            upsamplingKernel = zeros(strideHeight, strideWidth);
            upsamplingKernel(1,1) = 1;
            dLossdZUpsampled = kron(dLossdZ(:,:,c,n), upsamplingKernel);
            dLossdZUpsampled = dLossdZUpsampled(1:end-strideHeight+1,1:end-strideWidth+1);
            dLossdXTruncated = conv2(dLossdZUpsampled, ones(poolHeight, poolWidth),'full');
        else
            dLossdXTruncated = conv2(dLossdZ(:,:,c,n), ones(poolHeight, poolWidth),'full');
        end

        % Next, we expand the truncated version of dLossdX by padding it
        % with zeroes at the bottom and left. This is not only to account
        % for padding, but also to account situations where the parameters
        % do not "divide perfectly", and the output must therefore be zero
        % padded.
        truncatedHeight = size(dLossdXTruncated, 1);
        truncatedWidth = size(dLossdXTruncated, 2);
        XPaddedHeight = size(X,1) + padTop + padBottom;
        XPaddedWidth = size(X,2) + padLeft + padRight;
        dLossdXExpanded = zeros(XPaddedHeight, XPaddedWidth);
        dLossdXExpanded(1:truncatedHeight,1:truncatedWidth) = dLossdXTruncated;

        % If the input was padded, we need to undo the padding now.
        if (padTop > 0) || (padLeft > 0) || (padBottom > 0) || (padRight > 0)
            dLossdX(:,:,c,n) = dLossdXExpanded(1+padTop:end-padBottom, 1+padLeft:end-padRight);
        else
            dLossdX(:,:,c,n) = dLossdXExpanded;
        end
    end
end

% Divide by the pooling area
if(includePadding)
    dLossdX = dLossdX/(poolHeight*poolWidth);
else
    poolHeightsVector = poolHeight*ones(imageHeight,1);
    poolHeightsVector(1) = poolHeightsVector(1) - padTop;
    poolHeightsVector(end) = poolHeightsVector(end) - padBottom;
    poolWidthsVector = poolWidth*ones(1,imageWidth);
    poolWidthsVector(1) = poolWidthsVector(1) - padLeft;
    poolWidthsVector(end) = poolWidthsVector(end) - padRight;
    poolDivisors = poolHeightsVector * poolWidthsVector;
    dLossdX = bsxfun(@rdivide, dLossdX, poolDivisors);
end
end
