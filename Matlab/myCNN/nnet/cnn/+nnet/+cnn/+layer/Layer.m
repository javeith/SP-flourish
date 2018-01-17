classdef(Abstract) Layer <  matlab.mixin.CustomDisplay & nnet.cnn.layer.mixin.ScalarLayerDisplay & matlab.mixin.Heterogeneous
    % Layer   Interface for network layers
    %
    % To define the architecture of a network, create a vector of layers, e.g.,
    %
    %   layers = [
    %       imageInputLayer([28 28 3])
    %       convolution2dLayer([5 5], 10)
    %       reluLayer()
    %       fullyConnectedLayer(10)
    %       softmaxLayer()
    %       classificationLayer()
    %   ];
    %
    % See also nnet.cnn.layer, trainNetwork, imageInputLayer,
    % convolution2dLayer, reluLayer, maxPooling2dLayer,
    % averagePooling2dLayer, fullyConnectedLayer, softmaxLayer,
    % classificationLayer.
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties (Abstract, SetAccess = private)
        % Name   A name for the layer
        Name
    end
    
    properties(Access = protected)
        Version = 1
        PrivateLayer
    end
    
    methods(Hidden, Static)
        function layers = inferParameters(layers)
            iAssertFirstLayerIsAnInputLayer(layers);
            iAssertLastLayerIsAnOutputLayer(layers);
            iAssertPenultimateLayerIsSoftmaxLayer(layers);
            
            % Warn if duplicate names
            if iAnyDuplicateName( layers )
                warning(message('nnet_cnn:inferParameters:DuplicateNames'));
            end
            
            % Loop over the layers
            numLayers = numel(layers);
            inputSize = layers(1).InputSize;
            for i = 1:numLayers
                iAssertThatThisIsNotAnInputLayerAfterTheFirstLayer(layers(i), i);
                iAssertThatThisIsNotAnOutputLayerBeforeTheLastLayer(layers(i), i, numLayers);
                
                % Set default layer name if this is empty
                layers = iInferLayerName(layers, i);
                
                layers = iInferSize(layers, i, inputSize);
                inputSize = layers(i).PrivateLayer.forwardPropagateSize(inputSize);
            end
            
            % Append a unique suffix to duplicate names
            names = iGetLayerNames( layers );
            names = iMakeUniqueStrings( names );
            layers = iSetLayerNames( layers, names );
        end
        
        function internalLayers = getInternalLayers(layers)
            internalLayers = cell(numel(layers),1);
            for i = 1:numel(layers)
                internalLayers{i} = layers(i).PrivateLayer;
            end
        end
        
        function layers = createLayers(internalLayers)
            % createLayers Creates external layers from internal layers
            layers = cellfun( @iMapInternalToExternalLayer, internalLayers, 'UniformOutput', false );
            layers = vertcat( layers{:} );
        end
    end
    
    methods (Hidden)
        function displayAllProperties(this)
            proplist = properties( this );
            matlab.mixin.CustomDisplay.displayPropertyGroups( ...
                this, ...
                this.propertyGroupGeneral( proplist ) );
        end
    end
    
    methods(Abstract, Access = protected)
        [description, type] = getOneLineDisplay(layer)
    end
    
    methods (Sealed, Access = protected)
        function displayNonScalarObject(layers)
            % displayNonScalarObject   Display function for non scalar
            % objects
            if isvector( layers )
                header = sprintf( '  %s\n', getVectorHeader( layers ) );
                disp( header )
                layers.displayOneLines()
            else
                fprintf( '  %s', getArrayHeader( layers ) )
                layers.displayOnlyTypes()
            end
        end
        
        function displayEmptyObject(layers)
            displayEmptyObject@matlab.mixin.CustomDisplay(layers);
        end
    end
    
    methods (Static, Sealed, Access = protected)
        function defaultObject = getDefaultScalarElement() %#ok<STOUT>
            exception = iCreateExceptionFromErrorID('nnet_cnn:layer:Layer:NoDefaultScalarElement');
            throwAsCaller(exception);
        end
    end
    
    methods(Sealed, Access = private)
        function header = getVectorHeader( layers )
            % getVectorHeader   Return the header to be displayed for a
            % vector of layers
            sizeString = sprintf( '%dx%d', size( layers ) );
            className = matlab.mixin.CustomDisplay.getClassNameForHeader( layers );
            header = iGetStringMessage( ...
                'nnet_cnn:layer:Layer:VectorHeader', ...
                sizeString, ...
                className );
        end
        
        function header = getArrayHeader( layers )
            % getArrayHeader   Return the header to be displayed for an
            % array of layers of size >= 2
            
            sizeString = iSizeToString( size(layers) );
            className = matlab.mixin.CustomDisplay.getClassNameForHeader( layers );
            header = iGetStringMessage( ...
                'nnet_cnn:layer:Layer:ArrayHeader', ...
                sizeString, ...
                className );
        end
        
        function displayOneLines(layers)
            % displayOneLines   Display one line for each layer in the
            % vector
            names = iGetLayersNames( layers );
            maxNameLength = iMaxLength( names );
            [descriptions, types] = iGetOneLineDisplay( layers );
            maxTypeLength= iMaxLength( types );
            for idx=1:numel(layers)
                iDisplayOneLine( idx, ...
                    names{idx}, maxNameLength, ...
                    types{idx}, maxTypeLength, ...
                    descriptions{idx} )
            end
        end
        
        function displayOnlyTypes(layers)
            % displayOnlyTypes   Display types for each layer in the vector
            % in brackets. If there are more than 3 layers, the rest will
            % be substituted by '...'
            [~, types] = iGetOneLineDisplay( layers );
            fprintf(' (')
            fprintf('%s', types{1})
            numLayers = numel( layers );
            for idx=2:min( 3, numLayers )
                fprintf(', %s', types{idx})
            end
            if numLayers>3
                fprintf(', ...')
            end
            fprintf(')\n')
        end
    end
end

function sizeString = iSizeToString( sizeVector )
% iSizeToString   Convert a size vector into a formatted size string where
% each dimension is separated by 'x'.
sizeString = sprintf( '%d', sizeVector(1) );
for i=2:numel(sizeVector)
    sizeString = sprintf( '%sx%d', sizeString, sizeVector(i) );
end

end

function stringMessage = iGetStringMessage(id, varargin)
stringMessage = getString( message( id, varargin{:} ) );
end

function res = iMaxLength(strings)
% iMaxLength   Return the maximum string length of a cell array of strings
res = max(cellfun('length', strings));
end

function names = iGetLayersNames(layers)
% iGetLayersNames   Get the name for each layer.
names = arrayfun(@(x)iWrapApostrophe(x.Name), layers, 'UniformOutput', false);
end

function string = iWrapApostrophe(string)
string = ['''' string ''''];
end

function [descriptions, types] = iGetOneLineDisplay(layers)
[descriptions, types] = arrayfun( @(x)x.getOneLineDisplay, layers, 'UniformOutput', false );
end

function iDisplayOneLine(idx, name, maxNameWidth, type, maxTypeWidth, oneLineDescription)
% iDisplayOneLine   Display one line for a layer formatted in a table-like
% style.

fprintf( '    %2i   %-*s   %-*s   %s\n', ...
    idx, ...
    maxNameWidth, name, ...
    maxTypeWidth, type, ...
    oneLineDescription )

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
names = arrayfun(@(layer)layer.Name, layers, 'UniformOutput', false);
end

function layers = iSetLayerNames(layers, names)
for i=1:numel(layers)
    layers(i).PrivateLayer.Name = names{i};
end
end

function str = iMakeUniqueStrings(str)
str = matlab.lang.makeUniqueStrings( str );
end

function layers = iInferLayerName(layers, index)
% iInferLayerName   Assign a default name to the layer if its name is
% empty
if isempty(layers(index).PrivateLayer.Name)
    layers(index).PrivateLayer.Name = layers(index).PrivateLayer.DefaultName;
end
end

function layers = iInferSize(layers, index, inputSize)
if(~layers(index).PrivateLayer.HasSizeDetermined)
    % Infer layer size if its size is not determined
    try
        layers(index).PrivateLayer = layers(index).PrivateLayer.inferSize(inputSize);
    catch e
        throwWrongLayerSizeException( e, index );
    end
else
    % Otherwise make sure the size of the layer is correct
    iAssertCorrectSize( layers, index, inputSize );
end
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

function iAssertCorrectSize( layers, index, inputSize )
% iAssertCorrectSize   Check that layer size matches the input size,
% otherwise the architecture would be inconsistent.
if ~layers(index).PrivateLayer.isValidInputSize( inputSize )
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:WrongLayerSize', index);
    throwAsCaller(exception);
end
end

function externalLayer = iMapInternalToExternalLayer(internalLayer)

iInternalLayer = @(className)['nnet.internal.cnn.layer.' className];

switch class(internalLayer)
    case iInternalLayer('Convolution2D')
        externalLayer = nnet.cnn.layer.Convolution2DLayer(internalLayer);
    case iInternalLayer('CrossEntropy')
        externalLayer = nnet.cnn.layer.ClassificationOutputLayer(internalLayer);
    case iInternalLayer('Dropout')
        externalLayer = nnet.cnn.layer.DropoutLayer(internalLayer);
    case iInternalLayer('FullyConnected')
        externalLayer = nnet.cnn.layer.FullyConnectedLayer(internalLayer);
    case iInternalLayer('ImageInput')
        externalLayer = nnet.cnn.layer.ImageInputLayer(internalLayer);
    case iInternalLayer('LocalMapNorm2D')
        externalLayer = nnet.cnn.layer.CrossChannelNormalizationLayer(internalLayer);
    case iInternalLayer('MaxPooling2D')
        externalLayer = nnet.cnn.layer.MaxPooling2DLayer(internalLayer);
    case iInternalLayer('AveragePooling2D')
        externalLayer = nnet.cnn.layer.AveragePooling2DLayer(internalLayer);
    case iInternalLayer('ReLU')
        externalLayer = nnet.cnn.layer.ReLULayer(internalLayer);
    case iInternalLayer('Softmax')
        externalLayer = nnet.cnn.layer.SoftmaxLayer(internalLayer);
end
end

function iAssertFirstLayerIsAnInputLayer(layers)
if(~iThisIsAnInputLayer(layers(1)))
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:FirstLayerMustBeAnInputLayer');
    throwAsCaller(exception);
end
end

function tf = iThisIsAnInputLayer(layer)
tf = isa(layer.PrivateLayer,'nnet.internal.cnn.layer.ImageInput');
end

function iAssertPenultimateLayerIsSoftmaxLayer(layers)
if(~iThisIsASoftmaxLayer(layers(end-1)))
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:MissingSoftmaxLayer');
    throwAsCaller(exception);
end
end

function tf = iThisIsASoftmaxLayer(layer)
tf = isa(layer.PrivateLayer,'nnet.internal.cnn.layer.Softmax');
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
if(~iThisIsAnOutputLayer(layers(end)))
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:LastLayerMustBeAnOutputLayer');
    throwAsCaller(exception);
end
end

function tf = iThisIsAnOutputLayer(layer)
tf = isa(layer.PrivateLayer,'nnet.internal.cnn.layer.OutputLayer');
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