function externalConnections = internalToExternalConnections( internalConnections, layers )
% internalToExternalConnections   Convert internal connections to external
% connections
%
%   externalConnections = internalToExternalConnections(internalConnections, layers)
%   takes an internal connections matrix and a set of layers, and converts
%   them to an external layers table.
%
%   Inputs:
%       internalConnections - A matrix in the "internal" fomat. Each row is
%                             a connection. The matrix has four columns, 
%                             which represent (in order):
%                               - The index of the layer at the start of 
%                                 the connection.
%                               - The index of the output for the layer at 
%                                 the start of the connection.
%                               - The index of the layer at the end of the 
%                                 connection.
%                               - The index of the input for the layer at 
%                                 the end of the connection.
%       layers              - An array of external layers 
%                             (nnet.cnn.layer.Layer).
%
%   Output:
%       externalConnections - This format is a table. This is the format
%                             that users will see when they view the 
%                             "Connections" property of "DAGNetwork" or 
%                             "LayerGraph". The table has two columns:
%                               - Source: A cell array of char arrays.
%                               - Destination: A cell array of char arrays.

%   Copyright 2017 The MathWorks, Inc.

if(isempty(internalConnections))
    externalConnections = table( [],{},'VariableNames',{'Source','Destination'} );
else
    sourceList = {layers(internalConnections(:,1)).Name}.';
    destinationList = {layers(internalConnections(:,3)).Name}.';
    
    % Get the names for the layer inputs and outputs
    layerOutputList = iGetLayerOutputNames(layers, internalConnections(:,1:2));
    layerInputList = iGetLayerInputNames(layers, internalConnections(:,3:4));
    
    sourceList = iConcatenateCharArrayLists(sourceList, layerOutputList);
    destinationList = iConcatenateCharArrayLists(destinationList, layerInputList);
    
    externalConnections = table( sourceList,destinationList,'VariableNames',{'Source','Destination'} );
end

end

function layerOutputList = iGetLayerOutputNames(layers, sourceConnections)
numConnections = size(sourceConnections,1);
layerOutputList = cell(numConnections,1);
for i = 1:numConnections
    layerIndex = sourceConnections(i,1);
    outputIndexForThisLayer = sourceConnections(i,2);
    layerOutputList{i} = iGetLayerOutputName(layers(layerIndex), outputIndexForThisLayer);
end
end

function outputName = iGetLayerOutputName(layer, outputIndex)
layerClass = class(layer);
switch layerClass
    case 'nnet.cnn.layer.MaxPooling2DLayer'
        if layer.HasUnpoolingOutputs
            maxPoolingOutputNames = {'/out','/indices','/size'};
            outputName = maxPoolingOutputNames{outputIndex};
        else
            outputName = '';
        end
    otherwise
        outputName = '';
end
end

function layerInputList = iGetLayerInputNames(layers, destinationConnections)
numConnections = size(destinationConnections,1);
layerInputList = cell(numConnections,1);
for i = 1:numConnections
    layerIndex = destinationConnections(i,1);
    inputIndexForThisLayer = destinationConnections(i,2);
    layerInputList{i} = iGetLayerInputName(layers(layerIndex), inputIndexForThisLayer);
end
end

function inputName = iGetLayerInputName(layer, inputIndex)
layerClass = class(layer);
switch layerClass
    case 'nnet.cnn.layer.MaxUnpooling2DLayer'
        maxUnpoolingInputNames = {'/in','/indices','/size'};
        inputName = maxUnpoolingInputNames{inputIndex};
    case 'nnet.cnn.layer.AdditionLayer'
        inputName = ['/in' num2str(inputIndex)];
    case 'nnet.cnn.layer.DepthConcatenationLayer'
        inputName = ['/in' num2str(inputIndex)];
    case 'nnet.cnn.layer.Crop2DLayer'
        crop2dInputNames = {'/in', '/ref'};
        inputName = crop2dInputNames{inputIndex};
    otherwise
        inputName = '';
end
end

function outputList = iConcatenateCharArrayLists(firstList, secondList)
outputList = strcat(firstList, secondList);
end