function internalLayers = inferParameters(internalLayers)
% inferParameters   Infer parameters of an array of internal layers

%   Copyright 2017 The MathWorks, Inc.

iAssertFirstLayerIsAnInputLayer(internalLayers);
iAssertLastLayerIsAnOutputLayer(internalLayers);
iAssertLayerBeforeClassificationIsSoftmax(internalLayers);
iAssertLayerBeforeRegressionIsNotSoftmax(internalLayers);

% Warn if duplicate names
if iAnyDuplicateName( internalLayers )
    warning(message('nnet_cnn:inferParameters:DuplicateNames'));
end

% Loop over the layers
numLayers = numel(internalLayers);
inputSize = internalLayers{1}.InputSize;
for i = 1:numLayers
    iAssertThatThisIsNotAnInputLayerAfterTheFirstLayer(internalLayers{i}, i);
    iAssertThatThisIsNotAnOutputLayerBeforeTheLastLayer(internalLayers{i}, i, numLayers);
    
    % Set default layer name if this is empty
    internalLayers = iInferLayerName(internalLayers, i);
    
    internalLayers = iInferSize(internalLayers, i, inputSize);
    inputSize = internalLayers{i}.forwardPropagateSize(inputSize);
end

% Append a unique suffix to duplicate names
names = iGetLayerNames( internalLayers );
names = iMakeUniqueStrings( names );
internalLayers = iSetLayerNames( internalLayers, names );
end

function tf = iAnyDuplicateName(layers)
% iAnyDuplicateName   Return true if there is any duplicate name in layers
% array, excluding empty values
names = iGetLayerNames(layers);
% Exclude empty names
idx = cellfun(@isempty,names);
names(idx) = [];
uniqueNames = unique(names);
tf = numel(names)~=numel(uniqueNames);
end

function names = iGetLayerNames(layers)
names = cellfun(@(layer)layer.Name, layers, 'UniformOutput', false);
end

function layers = iSetLayerNames(layers, names)
for i=1:numel(layers)
    layers{i}.Name = names{i};
end
end

function str = iMakeUniqueStrings(str)
nonUnique = iGetNonUniqueStrings(str);
str = matlab.lang.makeUniqueStrings( str, nonUnique );
end

function nonUniqueStrings = iGetNonUniqueStrings(inputStrings)
[~,indicesForUnique] = unique(inputStrings);
nonUniqueStrings = inputStrings;
nonUniqueStrings(indicesForUnique) = [];
end

function layers = iInferLayerName(layers, index)
% iInferLayerName   Assign a default name to the layer if its name is
% empty
if isempty(layers{index}.Name)
    layers{index}.Name = layers{index}.DefaultName;
end
end

function layers = iInferSize(layers, index, inputSize)
if(~layers{index}.HasSizeDetermined)
    % Infer layer size if its size is not determined
    try
        layers{index} = layers{index}.inferSize(inputSize);
    catch e
        throwWrongLayerSizeException( e, index );
    end
end
% Additionally, make sure data of size inputSize are valid for the layer
iValidateInputSize( layers, index, inputSize );
end

function throwWrongLayerSizeException(e, index)
% throwWrongLayerSizeException   Throws a getReshapeDims:notSameNumel exception as
% a WrongLayerSize exception
if (strcmp(e.identifier,'MATLAB:getReshapeDims:notSameNumel'))
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:WrongLayerSize', index);
    throwAsCaller(exception)
else
    rethrow(e)
end
end

function iValidateInputSize( internalLayers, index, inputSize )
% iValidateInputSize   Check that the layer can propagate data of size
% inputSize, otherwise the architecture would be inconsistent

tf = internalLayers{index}.isValidInputSize( inputSize );
if ~tf
    % Throw a generic error to say that the output of layer index-1 was not
    % compatible with the input expected by layer index
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:IncompatibleLayers', index-1, index);
    throwAsCaller(exception);
end
end

function iAssertFirstLayerIsAnInputLayer(layers)
if(~iThisIsAnInputLayer(layers{1}))
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:FirstLayerMustBeAnInputLayer');
    throwAsCaller(exception);
end
end

function tf = iThisIsAnInputLayer(layer)
tf = isa(layer, 'nnet.internal.cnn.layer.ImageInput') || isa(layer,'nnet.internal.cnn.layer.SequenceInput');
end

function iAssertLayerBeforeClassificationIsSoftmax(layers)
if iThisIsAClassificationLayer(layers{end}) && ~iThisIsASoftmaxLayer(layers{end-1})
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:MissingSoftmaxLayer');
    throwAsCaller(exception);
end
end

function iAssertLayerBeforeRegressionIsNotSoftmax(layers)
if iThisIsARegressionLayer(layers{end}) && iThisIsASoftmaxLayer(layers{end-1})
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:SoftmaxLayerBeforeRegression');
    throwAsCaller(exception);
end
end

function tf = iThisIsASoftmaxLayer(layer)
tf = isa(layer,'nnet.internal.cnn.layer.Softmax');
end

function tf = iThisIsAClassificationLayer(layer)
tf = isa(layer,'nnet.internal.cnn.layer.ClassificationLayer');
end

function tf = iThisIsARegressionLayer(layer)
tf = isa(layer,'nnet.internal.cnn.layer.RegressionLayer');
end

function iAssertThatThisIsNotAnInputLayerAfterTheFirstLayer(layer, index)
if(iThisIsAnInputLayerAfterTheFirstLayer(layer, index))
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:LayersAfterTheFirstCannotBeInputLayers', index);
    throwAsCaller(exception);
end
end

function tf = iThisIsAnInputLayerAfterTheFirstLayer(layer, index)
tf = iThisIsAnInputLayer(layer) && (index > 1);
end

function iAssertLastLayerIsAnOutputLayer(layers)
if(~iThisIsAnOutputLayer(layers{end}))
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:LastLayerMustBeAnOutputLayer');
    throwAsCaller(exception);
end
end

function tf = iThisIsAnOutputLayer(layer)
tf = isa(layer,'nnet.internal.cnn.layer.OutputLayer');
end

function iAssertThatThisIsNotAnOutputLayerBeforeTheLastLayer(layer, index, numLayers)
if(iThisIsAnOutputLayerBeforeTheLastLayer(layer, index, numLayers))
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:LayersBeforeTheLastCannotBeOutputLayers', index);
    throwAsCaller(exception);
end
end

function tf = iThisIsAnOutputLayerBeforeTheLastLayer(layer, index, numLayers)
tf = iThisIsAnOutputLayer(layer) && (index < numLayers);
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(errorID, getString(message(errorID, varargin{:})));
end
