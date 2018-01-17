function layer = softmaxLayer( varargin )
% softmaxLayer   Softmax layer
%
%   layer = softmaxLayer() creates a softmax layer. This layer is 
%   useful for classification problems.
%
%   layer = softmaxLayer('PARAM1', VAL1) specifies optional parameter 
%   name/value pairs for creating the layer:
%
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   Example:
%       Create a softmax layer.
%
%       layer = softmaxLayer();
%
%   See also nnet.cnn.layer.SoftmaxLayer, classificationLayer, 
%   fullyConnectedLayer.

%   Copyright 2015 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = nnet.cnn.layer.SoftmaxLayer.parseInputArguments(varargin{:});

% Create an internal representation of a softmax layer.
internalLayer = nnet.internal.cnn.layer.Softmax(inputArguments.Name);

% Pass the internal layer to a  function to construct a user visible 
% softmax layer.
layer = nnet.cnn.layer.SoftmaxLayer(internalLayer);

end