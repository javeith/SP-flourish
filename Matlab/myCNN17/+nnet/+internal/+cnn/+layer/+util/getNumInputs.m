function numInputs = getNumInputs(layer)
% TODO: This should be a property of the layer, but to avoid changing that 
% class, we have created a function

% Copyright 2017 The MathWorks, Inc.

layerClass = class(layer);
switch layerClass
    case 'Some kind of layer that doesn''t exist yet'
        % This case just illustrates what you would need to do to specify 
        % that a layer has two inputs.
        numInputs = 2;
    case 'nnet.internal.cnn.layer.Addition'
        % This layer has a variable number of inputs, so we return NaN.
        numInputs = NaN;
    case 'nnet.internal.cnn.layer.Concatenation'
        % This layer has a variable number of inputs, so we return NaN.
        numInputs = NaN;
        
    case 'nnet.internal.cnn.layer.MaxUnpooling2D'
        numInputs = 3;
    case 'nnet.internal.cnn.layer.Crop2DLayer'
        numInputs = 2;
        
    otherwise
        % If the layer is not one of the layers mentioned above, it only 
        % has one input.
        numInputs = 1;
end
end