function [inputs,targets] = digitTestCellArrayData
% digitTestCellArrayData   Load the digit test set as cell array data
%
%   inputs      - Input data as a cell array of images.
%   targets     - Targets as a K-by-N matrix, where K is the number
%                 of classes and N is the number of images.

% Copyright 2016 The MathWorks, Inc.

digitsData = digitDatastore;

[~, testDigitsData] = splitEachLabel(digitsData,0.5);

[inputs, targets] = imds2cell( testDigitsData );

end