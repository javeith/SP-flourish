function dLossdX = convolveBackwardData2DCore( ...
    imageSize, W, dLossdZ, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    strideHeight, strideWidth)
% convolveBackwardData2DCore   Backpropagate through a
% convolutional layer to get the derivative with respect to the input.

% Copyright 2017 The MathWorks, Inc.

% The height and width of an input image. Note that this cannot be deduced
% from W and dLossdZ.
imageHeight = imageSize(1);
imageWidth = imageSize(2);

numInputChannels = size(W,3);
numExamples = size(dLossdZ,4);
numFilters = size(W,4);

outputHeight = size(dLossdZ,1);
outputWidth = size(dLossdZ,2);

% Upsample the input if the stride is greater than 1.
if (strideHeight > 1) || (strideWidth > 1)
    upsamplingKernel = zeros(strideHeight, strideWidth);
    upsamplingKernel(1,1) = 1;
    dLossdZUpsampled = zeros((outputHeight-1)*strideHeight+1, (outputWidth-1)*strideWidth+1, 'like', dLossdZ);
    for n = 1:numExamples
        for k = 1:numFilters
            dLossdZUpsampledOneChannel = kron(dLossdZ(:,:,k,n), upsamplingKernel);
            dLossdZUpsampled(:,:,k,n) = dLossdZUpsampledOneChannel(1:end-strideHeight+1,1:end-strideWidth+1);
        end
    end
    dLossdZ = dLossdZUpsampled;
end

% Transpose the weights, as we are doing backward convolution. We also need
% to flip the weights, because "convolveForward2D" actually does
% cross-correlation.
W = flip(W,2);
W = flip(W,1);
transposedWeights = permute(W, [1 2 4 3]);

% Calculate the padding we need for backward convolution. Backward
% convolution requires 'full' mode convolution (this means when the stride
% is 1, the output will be LARGER than the input).
filterHeight = size(W,1);
filterWidth = size(W,2);
verticalPadding = filterHeight - 1;
horizontalPadding = filterWidth - 1;

% Compute backward convolution via forward convolution.
dLossdX = nnet.internal.cnnhost.convolveBackward( ...
    dLossdZ, transposedWeights, ...
    verticalPadding, horizontalPadding, ...
    verticalPadding, horizontalPadding, ...
    strideHeight, strideWidth);

% Next, we expand dLossdX by padding it with zeroes at the bottom and left.
% This is not only to account for padding, but also to account for
% situations where the parameters do not "divide perfectly", and the output
% dLossdZ must therefore be zero padded.
truncatedHeight = size(dLossdX, 1);
truncatedWidth = size(dLossdX, 2);
XPaddedHeight = imageHeight + padTop + padBottom;
XPaddedWidth = imageWidth + padLeft + padRight;
dLossdXExpanded = zeros(XPaddedHeight, XPaddedWidth, numInputChannels, numExamples, 'like', dLossdZ);
dLossdXExpanded(1:truncatedHeight,1:truncatedWidth,:,:) = dLossdX;
dLossdX = dLossdXExpanded;

% If the input was padded, we need to undo the padding now.
if (padTop > 0) || (padLeft > 0) || (padBottom > 0) || (padRight > 0)
    dLossdX = dLossdX(1+padTop:end-padBottom,1+padLeft:end-padRight,:,:);
end

end

