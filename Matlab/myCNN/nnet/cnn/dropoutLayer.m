function layer = dropoutLayer( varargin )
% dropoutLayer   Dropout layer
%
%   layer = dropoutLayer() creates a dropout layer. During training, the 
%   dropout layer will randomly set input elements to zero with a 
%   probability of 0.5. This can be useful to prevent overfitting.
%
%   layer = dropoutLayer(probability) will create a dropout layer, where
%   probability is a number between 0 and 1 which specifies the probability
%   that an element will be set to zero. The default is 0.5.
%
%   layer = dropoutLayer(probability, 'PARAM1', VAL1) specifies optional 
%   parameter name/value pairs for creating the layer:
%
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   It is important to note that when creating a network, dropout will only
%   be used during training.
%
%   Example:
%       Create a dropout layer which will dropout roughly 40% of the input
%       elements.
%
%       layer = dropoutLayer(0.4);
%
%   See also nnet.cnn.layer.DropoutLayer, imageInputLayer, reluLayer.

%   Copyright 2015 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = nnet.cnn.layer.DropoutLayer.parseInputArguments(varargin{:});

% Create an internal representation of a dropout layer.
internalLayer = nnet.internal.cnn.layer.Dropout( ...
    inputArguments.Name, ...
    inputArguments.Probability);

% Pass the internal layer to a  function to construct a user visible 
% dropout layer.
layer = nnet.cnn.layer.DropoutLayer(internalLayer);

end