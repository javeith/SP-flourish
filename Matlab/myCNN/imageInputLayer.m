function layer = imageInputLayer(varargin)
% imageInputLayer   Image input layer
%
%   layer = imageInputLayer(inputSize) defines an image input layer.
%   inputSize is the size of the input images for the layer. It must be a
%   row vector of two or three numbers.
%
%   layer = imageInputLayer(inputSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%    'DataAugmentation'  Specify data augmentations to use during training
%                        as a string or cell array of strings. Valid
%                        augmentations are 'randcrop', 'randfliplr', or
%                        'none'.
%
%                        Default: 'none'
%
%    'Normalization'     Specify the data normalization to apply as a
%                        string. Valid values are 'zerocenter' or 'none'.
%                        Normalization is applied every time data is
%                        forward propagated through the input layer.
%
%                        Default: 'zerocenter'
%
%    'Name'              A name for the layer.
%
%                        Default: ''
%
%   Example:
%       Create an image input layer for 28-by-28 color images. At training
%       time, images will be flipped from left to right with a probability
%       of 50%.
%
%       layer = imageInputLayer([28 28 3], ...
%           'DataAugmentation', 'randfliplr');
%
%   See also nnet.cnn.layer.ImageInputLayer, convolution2dLayer, 
%   fullyConnectedLayer, maxPooling2dLayer.

%   Copyright 2015-2016 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = nnet.cnn.layer.ImageInputLayer.parseInputArguments(varargin{:});

normalization = iCreateTransforms(...
    inputArguments.Normalization, inputArguments.InputSize);

augmentations = iCreateTransforms(...
    inputArguments.DataAugmentation, inputArguments.InputSize);

% Create an internal representation of an image input layer.
internalLayer = nnet.internal.cnn.layer.ImageInput(...
    inputArguments.Name, ...
    inputArguments.InputSize, ...
    normalization, ...
    augmentations);

% Pass the internal layer to a function to construct a user visible image 
% input layer.
layer = nnet.cnn.layer.ImageInputLayer(internalLayer);
                                         
end

function tformarray = iCreateTransforms(type, imageSize)
type = cellstr(type);
tformarray = nnet.internal.cnn.layer.ImageTransform.empty();
for i = 1:numel(type)
    tnew = nnet.internal.cnn.layer.ImageTransformFactory.create(type{i}, imageSize);
    tformarray = [tformarray tnew]; %#ok<AGROW>
end
end
