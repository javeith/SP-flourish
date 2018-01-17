function [inputs,targets] = digitTrainCellArrayData
% digitTrainCellArrayData   Load the digit training set as cell array data
%
%   inputs      - Input data as a cell array of images.
%   targets     - Matrix of targets of size K-by-N, where K is the number
%                 of classes and N the number of images.

% Copyright 2016 The MathWorks, Inc.

digitsData = digitDatastore;

trainDigitsData = splitEachLabel( digitsData, 0.5 );

[inputs, targets] = imds2cell( trainDigitsData );

end