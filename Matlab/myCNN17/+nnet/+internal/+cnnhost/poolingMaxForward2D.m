function Z = poolingMaxForward2D(X, ...
    poolHeight, poolWidth, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    verticalStride, horizontalStride)
% poolingMaxForward2D   Forward max pooling on the host
%   Z = poolingMaxForward2D(X, poolHeight, poolWidth, verticalPad, horizontalPad, verticalStride, horizontalStride)
%   computes the max pooling Z of the input X using the pooling region 
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

% Placeholder for largest magnitude negative number
negMax = [];

% Apply padding to the images if necessary.
if (padTop > 0) || (padLeft > 0) || (padBottom > 0) || (padRight > 0)
    negMax = iMaximumNegativeValue(X);
    X = iPadArrayWithMaxNegativeValue(X, padTop, padLeft, padBottom, padRight, negMax);
end

% Allocate memory for the output.
Z = iAllocateArrayForOutput(X, poolHeight, poolWidth, verticalStride, horizontalStride);

% Perform max pooling.
pooledImageHeight = size(Z,1);
pooledImageWidth = size(Z,2);
for h = 1:pooledImageHeight
    for w = 1:pooledImageWidth
        startRow = (h-1)*verticalStride + 1;
        endRow = startRow + poolHeight - 1;
        startCol = (w-1)*horizontalStride + 1;
        endCol = startCol + poolWidth - 1;
        regionToPool = X(startRow:endRow, startCol:endCol, :, :);
        Z(h,w,:,:) = max(max(regionToPool,[],1),[],2);
    end
end

% Convert any NaNs to maximum negative value
iNaN = isnan(Z);
if any( iNaN(:) )
    if isempty( negMax)
        negMax = iMaximumNegativeValue(X);
    end
    Z( iNaN ) = negMax;
end
end

function Y = iPadArrayWithMaxNegativeValue(X, padTop, padLeft, padBottom, padRight, negMax)
paddedSize = size(X);
paddedSize(1) = paddedSize(1) + padTop + padBottom;
paddedSize(2) = paddedSize(2) + padLeft + padRight;
Y = negMax*ones(paddedSize, 'like', X);
imageTop = padTop + 1;
imageBottom = padTop + size(X,1);
imageLeft = padLeft + 1;
imageRight = padLeft + size(X,2);
Y(imageTop:imageBottom, imageLeft:imageRight, :, :) = X;
end

function Z = iAllocateArrayForOutput(X, poolHeight, poolWidth, verticalStride, horizontalStride)
paddedImageHeight = size(X,1);
paddedImageWidth = size(X,2);
numMaps = size(X,3);
numExamples = size(X,4);
pooledImageHeight = floor((paddedImageHeight-poolHeight)/verticalStride)+1;
pooledImageWidth = floor((paddedImageWidth-poolWidth)/horizontalStride)+1;
Z = zeros(pooledImageHeight, pooledImageWidth, numMaps, numExamples, 'like', X);
end

function Y = iMaximumNegativeValue(X)
Y = -realmax(class(X));
end