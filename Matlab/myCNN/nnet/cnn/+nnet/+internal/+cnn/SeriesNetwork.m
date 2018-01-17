classdef SeriesNetwork < nnet.internal.cnn.TrainableNetwork
    % SeriesNetwork   Class for a series convolutional neural network
    %
    %   A series network is always composed by an input layer, some middle
    %   layers, an output layer and a loss layer.
    %   Consistency of the layers and their conformity to this scheme has 
    %   to be checked outside the network.
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties
        % Layers    Layers of the networks
        %           (cell array of nnet.internal.cnn.layer.Layer)
        Layers = cell.empty;
    end
    
    properties (Dependent, SetAccess = private)
        % LearnableParameters    Learnable parameters of the networks
        %                        (vector of nnet.internal.cnn.layer.LearnableParameter)
        LearnableParameters
    end
    
    methods
        function this = SeriesNetwork(layers)
            % SeriesNetwork     Constructor for SeriesNetwork class
            %
            %   Create a series network with a cell array of
            %   nnet.internal.cnn.layer.Layer
            if nargin
                this.Layers = layers;
            end
        end
        
        function output = predict(this, data)
            % predict   Predict a response based on the input data
            indexOutputLayer = this.indexOutputLayer();
            output = activations(this, data, indexOutputLayer);
        end
        
        function output = activations(this, data, outputLayer)
            
            % Apply transforms to input. These transforms are applied on
            % the CPU prior to moving data to GPU.
            if iInputLayerHasTransforms(this.Layers{1})
                data = apply(this.Layers{1}.Transforms, data);
            end
            
            output = data;
            for currentLayer = 1:outputLayer
                output = this.Layers{currentLayer}.forward( output );
            end
        end
        
        function [gradients, loss, accuracy] = gradients(this, data, response)
            % gradients    Computes the gradients of the loss with respect
            % to the learnable parameters
            %
            % Input
            %   data - 4-D array containing the input data
            %   response - responses arranged by [ number of classes,
            %              number of observations ].
            %
            % Output
            %   gradients - cell array of gradients with one element for
            %               each learnable parameter
            %   loss - scalar double
            %   accuracy - scalar double
            outputLayerIndex = this.indexOutputLayer;
            data = gpuArray(data);
            
            % Do the forward pass and store the output from every layer
            [layerOutputs, memory] = this.forwardPropagation(data);
            
            [loss, accuracy] = iLossAndAccuracy( this.Layers{outputLayerIndex}, ...
                layerOutputs{outputLayerIndex}, response );
            
            % Backward propagation
            dxLayers = this.backwardPropagation(layerOutputs, response, memory);
            
            % Calculate the gradients with respect to the parameters
            gradients = {};
            for el = 2:outputLayerIndex-1 % Skip input and output layer
                theseGradients = this.Layers{el}.gradients(layerOutputs{el-1}, dxLayers{el+1});
                gradients = [gradients theseGradients];
            end
            
        end
        
        function this = updateLearnableParameters(this, deltas)
            % updateLearnableParameters   Update each learnable parameter
            % by subtracting a delta from it
            currentDelta = 1;
            for el = 1:this.indexOutputLayer
                for param = 1:numel(this.Layers{el}.LearnableParameters)
                    this.Layers{el}.LearnableParameters(param).Value = this.Layers{el}.LearnableParameters(param).Value + deltas{currentDelta};
                    currentDelta = currentDelta + 1;
                end
            end
        end
        
        function this = prepareNetworkForTraining(this)
            % prepareNetworkForTraining   Convert the network into a format
            % suitable for training
            for el = 1:this.indexOutputLayer
                this.Layers{el} = this.Layers{el}.prepareForTraining();
            end
        end
        
        function this = prepareNetworkForPrediction(this)
            % prepareNetworkForPrediction   Convert the network into a 
            % format suitable for prediction
            for el = 1:this.indexOutputLayer
                this.Layers{el} = this.Layers{el}.prepareForPrediction();
            end
        end
        
        function this = setupNetworkForHostPrediction(this)
            % setupNetworkForHostPrediction   Setup the network to perform
            % prediction on the host
            for el = 1:this.indexOutputLayer
                this.Layers{el} = this.Layers{el}.setupForHostPrediction();
            end
        end
        
        function this = setupNetworkForGPUPrediction(this)
            % setupNetworkForGPUPrediction   Setup the network to perform
            % prediction on the GPU
            for el = 1:this.indexOutputLayer
                this.Layers{el} = this.Layers{el}.setupForGPUPrediction();
            end
        end
    end
    
    methods
        function learnableParameters = get.LearnableParameters(this)
            learnableParameters = [];
            for el = 1:this.indexOutputLayer
                learnableParameters = [learnableParameters this.Layers{el}.LearnableParameters];
            end
        end
    end
    
    methods (Access = private)
        function indexOutputLayer = indexOutputLayer(this)
            % indexOutputLayer    Return what number is the output layer
            indexOutputLayer = numel(this.Layers);
        end
        
        function [layerOutputs, memory] = forwardPropagation(this, data)
            % forwardPropagation    Forward input data and returns a cell
            % array containing the output of each layer.
            %
            % Inputs
            %   data - a gpuArray containing the data
            % Outputs
            %   layerOutputs - a cell array containing the output of the
            %                  forward function on each layer
            indexOutputLayer = this.indexOutputLayer();
            layerOutputs = cell(indexOutputLayer,1);
            memory = cell(indexOutputLayer,1);
            
            [layerOutputs{1}, memory{1}] = this.Layers{1}.forward( data );
            for currentLayer = 2:indexOutputLayer
                [layerOutputs{currentLayer}, memory{currentLayer}] = this.Layers{currentLayer}.forward( layerOutputs{currentLayer-1} );
            end
        end
        
        function dxLayers = backwardPropagation(this, layerOutputs, response, memory)
            % backPropagation   Propagate the response from the last layer
            % to the first returing diffs between outputs and inputs
            
            indexOutputLayer = this.indexOutputLayer();
            
            dxLayers = cell(indexOutputLayer, 1);
            
            % Call backward loss on the output layer
            dxLayers{indexOutputLayer} = ...
                this.Layers{indexOutputLayer}.backwardLoss(layerOutputs{end}, response);
            
            % Call backward on every other layer, except the first since
            % its delta will be empty
            for el = indexOutputLayer-1:-1:2
                dxLayers{el} = this.Layers{el}.backward(...
                    layerOutputs{el-1}, layerOutputs{el}, dxLayers{el+1}, memory{el});
            end
        end
    end
end

function tf = iInputLayerHasTransforms(layer)
% only input layers have transforms.
tf = isa(layer, 'nnet.internal.cnn.layer.ImageInput');
end

function [loss,accuracy] = iLossAndAccuracy(outputLayer, output, response)
% iLossAndAccuracy   Loss and accuracy of the current network

% Calculate the loss from the output layer
loss = outputLayer.forwardLoss(output, response);

% Calculate the accuracy on the current mini-batch
accuracy = iComputeAccuracy(output, response);
end

function accuracy = iComputeAccuracy(y, t)
% y is the network output and t is the network response. They are expected
% to be of size 1 x 1 x numClasses x numObservations.
[~, yIntegers] = max(y, [], 3);
[~, tIntegers] = max(t, [], 3);
numObservations = numel(yIntegers);
% numel(tIntegers) == numObservations as well, or next part will error

accuracy = gather(100 * (sum(yIntegers == tIntegers)/numObservations));
end
