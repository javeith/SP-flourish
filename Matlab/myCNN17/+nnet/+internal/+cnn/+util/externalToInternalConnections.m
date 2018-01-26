function internalConnections = externalToInternalConnections( externalConnections, layers )
% externalToInternalConnections   Convert external connections to internal
% connections.
%
%   internalConnections = externalToInternalConnections( externalConnections, layers )
%   takes an external connections table and a set of external layers, and
%   converts them to an internal connections matrix.
%
%   Inputs:
%       externalConnections - This format is a table. This is the format
%                             that users will see when they view the
%                             "Connections" property of "DAGNetwork" or
%                             "LayerGraph". The table has two columns:
%                               - Source: A cell array of char arrays.
%                               - Destination: A cell array of char arrays.
%       layers              - An array of external layers
%                             (nnet.cnn.layer.Layer).
%
%   Output:
%       internalConnections - A connections matrix in the "internal"
%                             format. Each row is a connection. The matrix
%                             has four columns, which represent (in order):
%                               - The index of the layer at the start of
%                                 the connection.
%                               - The index of the output for the layer at
%                                 the start of the connection.
%                               - The index of the layer at the end of the
%                                 connection.
%                               - The index of the input for the layer at
%                                 the end of the connection.

%   Copyright 2017 The MathWorks, Inc.

if isempty(externalConnections)
    internalConnections = zeros(0,4);
else
    % externalConnections.Source and externalConnections.Destination are
    % cell arrays of character vectors with elements that may look like
    % this:
    %
    %     {'addition_1/in1'}
    %     {'relu_3'        }
    %     {'conv_4'        }
    %     {'addition_2/in2'}
    %     {'batchnorm_4'   }
    %     {'relu_4'        }
    %
    % The character '/' is the delimiter that separates a layer name from a
    % port name. For example, in 'addition_1/in1', 'addition_1' is the
    % layer name and 'in1' is the port name. We assume that the last
    % character in port name when converted to double represents the port
    % number. These port numbers are the output ports for elements of
    % externalConnections.Source and input ports for elements of
    % externalConnections.Destination. For elements such as 'conv_4' where
    % delimiter '/' is absent, the port number is assumed to be 1.
    %
    % There are a few exceptions to this rule:
    % o For the MaxPooling2DLayer, output ports with names 'out', 'indices'
    % and 'size' map to port numbers 1, 2 and 3 respectively.
    % o For the MaxUnpooling2DLayer, input ports with names 'in', 'indices'
    % and 'size' map to port numbers 1, 2 and 3 respectively.
    
    % Get external layer names.
    externalLayerNames = arrayfun(@(x) x.Name, layers, 'UniformOutput',false);
    
    % Extract layer names and port names for each connection.
    [sourceLayerNames, sourcePortNames] = iExtractLayerAndPortNames(externalConnections.Source);
    [destinationLayerNames, destinationPortNames] = iExtractLayerAndPortNames(externalConnections.Destination);
    
    % Convert source and destination layer names to layer IDs. If a source
    % or destination layer name matches externalLayerNames{i} then that
    % source or destination layer gets the ID i.
    sourceLayerIDs = iConvertLayerNamesToIDs(sourceLayerNames, externalLayerNames);
    destinationLayerIDs = iConvertLayerNamesToIDs(destinationLayerNames, externalLayerNames);
    
    % Convert source and destination port names to port numbers.
    sourcePortNumbers = iConvertSourcePortNamesToNumbers(sourcePortNames, sourceLayerIDs, layers);
    destinationPortNumbers = iConvertDestinationPortNamesToNumbers(destinationPortNames, destinationLayerIDs, layers);
    
    % Create matrix representing internal connections.
    internalConnections = [sourceLayerIDs, sourcePortNumbers,...
        destinationLayerIDs, destinationPortNumbers];
end
end

function [layerNames,portNames] = iExtractLayerAndPortNames(layerAndPortNames)
delimiter = '/';
splitLayerAndPortNames = cellfun(@(x) strsplit(x,delimiter), layerAndPortNames,'UniformOutput',false);
havePortNames = cellfun(@(x) numel(x)==2, splitLayerAndPortNames);
layerNames = cellfun(@(x) x{1}, splitLayerAndPortNames,'UniformOutput',false);
portNames = cell(numel(layerNames),1);
portNames(havePortNames) = cellfun(@(x) x{2}, splitLayerAndPortNames(havePortNames),'UniformOutput',false);
end

function layerIDs = iConvertLayerNamesToIDs(layerNames, externalLayerNames)
layerIDs = zeros(numel(layerNames),1);
for i = 1:numel(externalLayerNames)
    currentLayerName = externalLayerNames{i};
    indicesMatchingCurrentLayerName = cellfun(@(x) strcmp(x,currentLayerName), layerNames);
    layerIDs(indicesMatchingCurrentLayerName) = i;
end
end

function sourcePortNumbers = iConvertSourcePortNamesToNumbers(sourcePortNames, sourceLayerIDs, layers)
haveSourcePortNames = true;
sourcePortNumbers = iConvertPortNamesToNumbers(sourcePortNames, sourceLayerIDs, layers, haveSourcePortNames);
end

function destinationPortNumbers = iConvertDestinationPortNamesToNumbers(destinationPortNames, destinationLayerIDs, layers)
haveSourcePortNames = false;
destinationPortNumbers = iConvertPortNamesToNumbers(destinationPortNames, destinationLayerIDs, layers, haveSourcePortNames);
end

function portNumbers = iConvertPortNamesToNumbers(portNames, layerIDs, layers, haveSourcePortNames)
numConnections = numel(portNames);
portNumbers = zeros(numConnections,1);
for i = 1:numConnections
    currentPortName = portNames{i};
    currentLayer = layers(layerIDs(i));
    if isempty(currentPortName)
        portNumbers(i) = 1;
    else
        portNumbers(i) = iGetPortNumber(currentPortName, currentLayer, haveSourcePortNames);
    end
end
end

function portNumber = iGetPortNumber(portName, layer, haveSourcePortName)
if haveSourcePortName
    portNumber = iGetSourcePortNumber(portName, layer);
else
    portNumber = iGetDestinationPortNumber(portName, layer);
end
end

function sourcePortNumber = iGetSourcePortNumber(sourcePortName, layer)
layerClass = class(layer);
switch layerClass
    case 'nnet.cnn.layer.MaxPooling2DLayer'
        if layer.HasUnpoolingOutputs
            maxPoolingOutputNames = {'out','indices','size'};
            [~,sourcePortNumber] = ismember(sourcePortName, maxPoolingOutputNames);
        else
            sourcePortNumber = 1;
        end
    otherwise
        sourcePortNumber = str2double(regexp(sourcePortName, '[0-9]+$', 'match'));
end
end

function destinationPortNumber = iGetDestinationPortNumber(destinationPortName, layer)
layerClass = class(layer);
switch layerClass
    case 'nnet.cnn.layer.MaxUnpooling2DLayer'
        maxUnpoolingInputNames = {'in','indices','size'};
        [~,destinationPortNumber] = ismember(destinationPortName, maxUnpoolingInputNames);
    case 'nnet.cnn.layer.Crop2DLayer'
        crop2dInputNames = {'in','ref'};
        [~,destinationPortNumber] = ismember(destinationPortName, crop2dInputNames);
    otherwise
        destinationPortNumber = str2double(regexp(destinationPortName, '[0-9]+$', 'match'));
end
end