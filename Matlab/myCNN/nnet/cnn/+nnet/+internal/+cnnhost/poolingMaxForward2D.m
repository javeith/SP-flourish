function Z = poolingMaxForward2D(X, ...
    poolHeight, poolWidth, ...
    verticalPad, horizontalPad, ...
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
    X = iPadArrayWithMaxNegativeValue(X, verticalPad, horizontalPad);
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
end

function Y = iPadArrayWithMaxNegativeValue(X, verticalPad, horizontalPad)
paddedSize = size(X);
paddedSize(1) = paddedSize(1) + 2*verticalPad;
paddedSize(2) = paddedSize(2) + 2*horizontalPad;
maxNegativeValue = -realmax(class(X));
Y = maxNegativeValue*ones(paddedSize, 'like', X);
imageTop = verticalPad + 1;
imageBottom = verticalPad + size(X,1);
imageLeft = horizontalPad + 1;
imageRight = horizontalPad + size(X,2);
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