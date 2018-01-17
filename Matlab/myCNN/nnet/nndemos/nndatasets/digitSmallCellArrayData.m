function [inputs, targets] = digitSmallCellArrayData
% digitSmallCellArrayData   Small synthetic handwritten digit dataset
%   [inputs, targets] = digitsmall_dataset will load a dataset of synthetic
%   handwritten digits, where inputs will be a 1-by-500 cell array, with
%   each cell containing a 28-by-28 matrix representing a synthetic image
%   of a handwritten digit, and targets will be a 10-by-500 matrix
%   containing the labels for the images in 1-of-K format. This dataset is
%   a subset of digittrain_dataset.

% Copyright 2016 The MathWorks, Inc.

digitsData = digitDatastore;

trainDigitsData = splitEachLabel( digitsData, 50 );

[inputs, targets] = imds2cell( trainDigitsData );

end