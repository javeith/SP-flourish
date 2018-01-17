function digitData = digitDatastore
% digitData   Load the digit dataset image datastore
%
%   digitData       - An image datastore containing the digit dataset

% Copyright 2016 The MathWorks, Inc.

digitDatasetPath = fullfile( matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'DigitDataset' );

digitData = imageDatastore( digitDatasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames' );

end