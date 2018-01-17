function [inputs,targets] = digitTrain4DArrayData
% digitTrain4DArrayData   Load the digit training set as 4-D array data
%
%   inputs      - Input data as an H-by-W-by-C-by-N array, where H is the
%                 height and W is the width of the images, C is the number
%                 of channels, and N is the number of images.
%   targets     - Categorical vector containing the labels for each 
%                 observation.

% Copyright 2016 The MathWorks, Inc.

digitsData = digitDatastore;

minSetCount = min( digitsData.countEachLabel{:,2} );

trainDigitsData = splitEachLabel( digitsData, round(minSetCount/2) );

[inputs, targets] = imds2array(trainDigitsData);

end