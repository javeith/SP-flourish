function layer = averagePooling2dLayer( varargin )
% averagePooling2dLayer   Average pooling layer
%
%   layer = averagePooling2dLayer(poolSize) creates a layer that performs
%   average pooling. An average pooling layer divides the input into
%   rectangular pooling regions, and outputs the average of each region.
%   poolSize specifies the width and height of a pooling region. It can be
%   a scalar, in which case the pooling regions will have the same width
%   and height, or a vector [h w] where h specifies the height and w
%   specifies the width. Note that if the 'Stride' dimensions are less than
%   the respective pool dimensions, then the pooling regions will overlap.
%
%   layer = averagePooling2dLayer(poolSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...) 
%   specifies optional parameter name/value pairs for creating the layer:
%
%       'Stride'                  - The step size for traversing the input
%                                   vertically and horizontally. This can
%                                   be a scalar, in which case the same
%                                   value is used for both dimensions, or
%                                   it can be a vector [u v] where u is the
%                                   vertical stride, and v is the
%                                   horizontal stride. Values that are 
%                                   greater than 1 can be used to 
%                                   down-sample the input. The default is
%                                   [1 1].
%       'Padding'                 - The padding applied to the input
%                                   vertically and horizontally. This can
%                                   be a scalar, in which case the same
%                                   padding is applied vertically and
%                                   horizontally, or a vector [a b] where
%                                   a is the padding applied to the top
%                                   and bottom of the input, and b is the
%                                   padding applied to the left and right. 
%                                   Note that the padding dimensions must
%                                   be less than the pooling region
%                                   dimensions. The default is [0 0].
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   Example 1:
%       Create an average pooling layer with non-overlapping pooling
%       regions, which downsamples by a factor of 2:
%
%       layer = averagePooling2dLayer(2, 'Stride', 2);
%
%   Example 2:
%       Create an average pooling layer with overlapping pooling regions
%       and padding for the top and bottom of the input:
%
%       layer = averagePooling2dLayer(3, 'Stride', 2, 'Padding', [1 0]);
%
%   See also nnet.cnn.layer.AveragePooling2DLayer, maxPooling2dLayer,
%   convolution2dLayer.

%   Copyright 2015-2016 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = nnet.cnn.layer.AveragePooling2DLayer.parseInputArguments(varargin{:});

% Create an internal representation of an average pooling layer.
internalLayer = nnet.internal.cnn.layer.AveragePooling2D( ...
    inputArguments.Name, ...
    inputArguments.PoolSize, ...
    inputArguments.Stride, ...
    inputArguments.Padding);

% Pass the internal layer to a function to construct a user visible
% average pooling layer
layer = nnet.cnn.layer.AveragePooling2DLayer(internalLayer);

end