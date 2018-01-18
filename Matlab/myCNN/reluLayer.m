function layer = reluLayer(varargin)
% reluLayer   Rectified linear unit (ReLU) layer
%
%   layer = reluLayer() creates a rectified linear unit layer. This type of
%   layer performs a simple thresholding operation, where any input value
%   less than zero will be set to zero.
%
%   layer = reluLayer('PARAM1', VAL1) specifies optional parameter 
%   name/value pairs for creating the layer:
%
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   Example:
%       Create a rectified linear unit layer.
%
%       layer = reluLayer();
%
%   See also nnet.cnn.layer.ReLULayer, convolution2dLayer, 
%   fullyConnectedLayer.

%   Copyright 2015 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = nnet.cnn.layer.ReLULayer.parseInputArguments(varargin{:});

% Create an internal representation of a ReLU layer.
internalLayer = nnet.internal.cnn.layer.ReLU(inputArguments.Name);

% Pass the internal layer to a  function to construct a user visible ReLU 
% layer.
layer = nnet.cnn.layer.ReLULayer(internalLayer);

end