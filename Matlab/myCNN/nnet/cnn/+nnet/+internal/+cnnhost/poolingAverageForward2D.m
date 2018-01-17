function Z = poolingAverageForward2D(X, ...
    poolHeight, poolWidth, ...
    verticalPad, horizontalPad, ...
    verticalStride, horizontalStride)
% poolingAverageForward2D   Forward average pooling on the host
%   Z = poolingAverageForward2D(X, poolHeight, poolWidth, verticalPad, horizontalPad, verticalStride, horizontalStride)
%   computes the average pooling Z of the input X using the pooling region 
%   size defined by poolHeight and poolWidth. Padding size is set with 
%   verticalPad and horizontalPad, and the vertical and horizontal stride 
%   are set with verticalStride and horizontalStride.
%
%   Inputs:
%   X - Input channels for a set of images. A (H)x(W)x(C)x(N) array.
%   poolHeight - The height of each pooling region
%   poolWidth - The width of each pooling region
%   verticalPad - The vertical padding (applied to the top and bottom).
%   horizontalPad - The horizontal padding (applied to the left and right).
%   verticalStride - The vertical stride.
%   horizontalStride - The horizontal stride.
%
%   Output:
%   Z - The output feature channels for the images. A
%       floor((H + 2*verticalPad - poolHeight)/verticalStride + 1) x
%       floor((W + 2*horizontalPad - poolWidth)/horizontalStride + 1) x
%       (C) x (N) array.

%   Copyright 2016 The MathWorks, Inc.

% Apply padding to the images if necessary.
if((verticalPad > 0)||(horizontalPad > 0))
    X = iPadArray(X, verticalPad, horizontalPad);
end

% Perform average pooling, ignoring the stride. (stride can be accounted 
% for by downsampling this result).
Z = iAveragePoolingWithoutStride(X, poolHeight, poolWidth);

% Downsample the output to account for stride.
Z = Z(1:verticalStride:end, 1:horizontalStride:end, :, :);

% Normalize the result
Z = Z/(poolWidth*poolHeight);
end

function Y = iPadArray(X, verticalPad, horizontalPad)
paddedSize = size(X);
paddedSize(1) = paddedSize(1) + 2*verticalPad;
paddedSize(2) = paddedSize(2) + 2*horizontalPad;
Y = zeros(paddedSize, 'like', X);
imageTop = verticalPad + 1;
imageBottom = verticalPad + size(X,1);
imageLeft = horizontalPad + 1;
imageRight = horizontalPad + size(X,2);
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