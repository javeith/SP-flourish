function Z = poolingAverageForward2D(X, ...
    poolHeight, poolWidth, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    verticalStride, horizontalStride)
% poolingAverageForward2D   Forward average pooling on the host
%   Z = poolingAverageForward2D(X, poolHeight, poolWidth, padTop, padLeft, padBottom, padRight, verticalStride, horizontalStride)
%   computes the average pooling Z of the input X using the pooling region 
%   size defined by poolHeight and poolWidth. Padding size is set with 
%   verticalPad and horizontalPad, and the vertical and horizontal stride 
%   are set with verticalStride and horizontalStride.
%
%   Inputs:
%   X - Input channels for a set of images. A (H)x(W)x(C)x(N) array.
%   poolHeight - The height of each pooling region
%   poolWidth - The width of each pooling region
%   padTop - Padding on the top.
%   padLeft - Padding on the left.
%   padBottom - Padding on the bottom.
%   padRight - Padding on the right.
%   verticalStride - The vertical stride.
%   horizontalStride - The horizontal stride.
%
%   Output:
%   Z - The output feature channels for the images. A
%       floor((H + padTop + padBottom - poolHeight)/verticalStride + 1) x
%       floor((W + padLeft + padRight - poolWidth)/horizontalStride + 1) x
%       (C) x (N) array.

%   Copyright 2016-2017 The MathWorks, Inc.

% Apply padding to the images if necessary.
if (padTop > 0) || (padLeft > 0) || (padBottom > 0) || (padRight > 0)
    X = iPadArray(X, padTop, padLeft, padBottom, padRight);
end

% Perform average pooling, ignoring the stride. (stride can be accounted 
% for by downsampling this result).
Z = iAveragePoolingWithoutStride(X, poolHeight, poolWidth);

% Downsample the output to account for stride.
Z = Z(1:verticalStride:end, 1:horizontalStride:end, :, :);

% Normalize the result
Z = Z/(poolWidth*poolHeight);
end

function Y = iPadArray(X, padTop, padLeft, padBottom, padRight)
paddedSize = size(X);
paddedSize(1) = paddedSize(1) + padTop + padBottom;
paddedSize(2) = paddedSize(2) + padLeft + padRight;
Y = zeros(paddedSize, 'like', X);
imageTop = padTop + 1;
imageBottom = padTop + size(X,1);
imageLeft = padLeft + 1;
imageRight = padLeft + size(X,2);
Y(imageTop:imageBottom, imageLeft:imageRight, :, :) = X;
end

function Z = iAveragePoolingWithoutStride(X, poolHeight, poolWidth)

% Define a filter (rotation is not needed because it is symmetric).
meanFilter = ones(poolHeight, poolWidth, 'like', X);

% Allocate memory for the (un-downsampled) output.
Z = iAllocateArrayForOutputWithoutStride(X, meanFilter);

% Perform average pooling through convolution.
numExamples = size(X,4);
numChannels = size(X,3);
for n = 1:numExamples
    for c = 1:numChannels
        Z(:,:,c,n) = conv2(X(:,:,c,n), meanFilter, 'valid');
    end
end
end

function Z = iAllocateArrayForOutputWithoutStride(X, W)
paddedSize = size(X);
filterSize = size(W);
convolvedImageHeightWithoutStride = paddedSize(1) - filterSize(1) + 1;
convolvedImageWidthWithoutStride = paddedSize(2) - filterSize(2) + 1;
numOutputChannels = size(X,3);
numExamples = size(X,4);
Z = zeros(convolvedImageHeightWithoutStride, ...
    convolvedImageWidthWithoutStride, ...
    numOutputChannels, numExamples, 'like', X);
end