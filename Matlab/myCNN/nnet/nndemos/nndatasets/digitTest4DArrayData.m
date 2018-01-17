function [inputs,targets] = digitTest4DArrayData
% digitTest4DArrayData   Load the digit test set as 4-D array data
%
%   inputs      - Input data as a H-by-W-by-C-by-N array, where H is the
%                 height of the images, W is the width, C the number of
%                 channels and N the number of images.
%   targets     - Categorical vector containing the labels for each 
%                 observation.

% Copyright 2016 The MathWorks, Inc.

digitsData = digitDatastore;

minSetCount = min( digitsData.countEachLabel{:,2} );

[~, testDigitsData] = splitEachLabel( digitsData, round(minSetCount/2) );

[inputs, targets] = imds2array( testDigitsData );

end