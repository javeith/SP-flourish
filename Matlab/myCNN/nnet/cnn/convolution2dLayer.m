function layer = convolution2dLayer( varargin )
% convolution2dLayer   2D convolution layer for Convolutional Neural Networks
%
%   layer = convolution2dLayer(filterSize, numFilters) creates a layer
%   for 2D convolution. filterSize specifies the height and width of the
%   filters. It can be a scalar, in which case the filters will have the
%   same height and width, or a vector [h w] where h specifies the height
%   for the filters, and w specifies the width. numFilters specifies the
%   number of filters.
% 
%   layer = convolution2dLayer(filterSize, numFilters, 'PARAM1', VAL1, 'PARAM2', VAL2, ...) 
%   specifies optional parameter name/value pairs for creating the layer:
%
%       'Stride'                  - The step size for traversing the input
%                                   vertically and horizontally. This can
%                                   be a scalar, in which case the same
%                                   value is used for both dimensions, or
%                                   it can be a vector [u v] where u is the
%                                   vertical stride, and v is the
%                                   horizontal stride. The default is 
%                                   [1 1].
%       'Padding'                 - The padding applied to the input
%                                   vertically and horizontally. This can
%                                   be a scalar, in which case the same
%                                   padding is applied vertically and
%                                   horizontally, or a vector [a b] where
%                                   a is the padding applied to the top
%                                   and bottom of the input, and b is the
%                                   padding applied to the left and right. 
%                                   The default is [0 0].
%       'NumChannels'             - The number of channels for each filter.
%                                   If a value of 'auto' is passed in, the
%                                   correct value for this parameter will
%                                   be inferred at training time. The
%                                   default is 'auto'.
%       'WeightLearnRateFactor'   – A number that specifies multiplier for
%                                   the learning rate of the weights. The
%                                   default is 1.
%       'BiasLearnRateFactor'     - A number that specifies a multiplier
%                                   for the learning rate for the biases. 
%                                   The default is 1.
%       'WeightL2Factor'          - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   weights. The default is 1.
%       'BiasL2Factor'            - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   biases. The default is 1.
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   Example 1:
%       Create a convolutional layer with 96 filters that have a height and
%       width of 11, and use a stride of 4 in the horizontal and vertical 
%       directions.
%
%       layer = convolution2dLayer(11, 96, 'Stride', 4);
%
%   Example 2:
%       Create a convolutional layer with 32 filters that have a height and
%       width of 5. Pad the input image with 2 pixels along its border. Set
%       the learning rate factor for the bias to 2. Manually initialize the
%       weights from a Gaussian with standard deviation 0.0001.
%
%       layer = convolution2dLayer(5, 32, 'Padding', 2, 'BiasLearnRateFactor', 2);
%       layer.Weights = randn([5 5 3 32])*0.0001;
%
%   See also nnet.cnn.layer.Convolution2DLayer, maxPooling2dLayer, 
%   averagePooling2dLayer.

%   Copyright 2015-2016 The MathWorks, Inc.

% Parse the input arguments.
args = nnet.cnn.layer.Convolution2DLayer.parseInputArguments(varargin{:});

% Create an internal representation of a convolutional layer.
internalLayer = nnet.internal.cnn.layer.Convolution2D(args.Name, ...
                                                      args.FilterSize, ...
                                                      args.NumChannels, ...
                                                      args.NumFilters, ...
                                                      args.Stride, ...
                                                      args.Padding);

internalLayer.Weights.L2Factor = args.WeightL2Factor;
internalLayer.Weights.LearnRateFactor = args.WeightLearnRateFactor;

internalLayer.Bias.L2Factor = args.BiasL2Factor;
internalLayer.Bias.LearnRateFactor = args.BiasLearnRateFactor;

% Pass the internal layer to a function to construct a user visible
% convolutional layer.
layer = nnet.cnn.layer.Convolution2DLayer(internalLayer);

end