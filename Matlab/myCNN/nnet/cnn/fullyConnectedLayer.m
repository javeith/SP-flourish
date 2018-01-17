function layer = fullyConnectedLayer( varargin )
% fullyConnectedLayer   Fully connected layer
%
%   layer = fullyConnectedLayer(outputSize) creates a fully connected
%   layer. outputSize specifies the size of the output for the layer. A
%   fully connected layer will multiply the input by a matrix and then add
%   a bias vector.
%
%   layer = fullyConnectedLayer(outputSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...) 
%   specifies optional parameter name/value pairs for creating the layer:
%
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
%   Example 1:
%       Create a fully connected layer with an output size of 10, and an
%       input size that will be determined at training time.
%
%       layer = fullyConnectedLayer(10);
%
%   See also nnet.cnn.layer.FullyConnectedLayer, convolution2dLayer, 
%   reluLayer.

%   Copyright 2015-2016 The MathWorks, Inc.

% Parse the input arguments.
args = nnet.cnn.layer.FullyConnectedLayer.parseInputArguments(varargin{:});

% Create an internal representation of a fully connected layer.
internalLayer = nnet.internal.cnn.layer.FullyConnected( ...
    args.Name, ...
    args.InputSize, ...
    args.OutputSize);

internalLayer.Weights.L2Factor = args.WeightL2Factor;
internalLayer.Weights.LearnRateFactor = args.WeightLearnRateFactor;

internalLayer.Bias.L2Factor = args.BiasL2Factor;
internalLayer.Bias.LearnRateFactor = args.BiasLearnRateFactor;

% Pass the internal layer to a function to construct a user visible
% fully connected layer.
layer = nnet.cnn.layer.FullyConnectedLayer(internalLayer);

end