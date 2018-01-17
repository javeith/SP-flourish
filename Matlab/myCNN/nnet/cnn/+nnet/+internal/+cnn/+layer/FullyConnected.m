classdef FullyConnected < nnet.internal.cnn.layer.Layer
    % FullyConnected   Implementation of the fully connected layer
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties
        % LearnableParameters   The learnable parameters for this layer
        %   This layer has two learnable parameters, which are the weights
        %   and the bias.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty()

        % Name (char array)   A name for the layer
        Name
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'fc'
    end
            
    properties(SetAccess = private)        
        % InputSize   Input size of the layer
        %   The input size of the fully connected layer. Note that the
        %   internal layers deal with 3D observations, and so this input
        %   size will be 3D. This will be empty if the input size has not
        %   been set yet.
        InputSize
        
        % NumNeurons  (scalar integer)   Number of neurons for the layer
        NumNeurons
    end
    
    properties(Access = private)
        ExecutionStrategy
    end
    
    properties (Dependent)
        % Weights   The weights for the layer
        Weights
        
        % Bias   The bias vector for the layer
        Bias
    end
    
    properties (Dependent, SetAccess = private)
        % HasSizeDetermined   Specifies if all size parameters are set
        %   If the input size has not been determined, then this will be
        %   set to false, otherwise it will be true.
        HasSizeDetermined        
    end
    
    properties (Constant, Access = private)
        % WeightsIndex   Index of the Weights in the LearnableParameter vector
        WeightsIndex = 1;
        
        % BiasIndex   Index of the Bias in the LearnableParameter vector
        BiasIndex = 2;
    end
    
    methods
        function this = FullyConnected(name, inputSize, numNeurons)
            this.Name = name;
            
            % Set hyperparameters
            this.NumNeurons = numNeurons;
            this.InputSize = inputSize;
            
            this.Weights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.FullyConnectedGPUStrategy();
        end
        
        function [Z, memory] = forward(this, X)
            [Z, memory] = this.ExecutionStrategy.forward( ...
                X, this.Weights.Value, this.Bias.Value);
        end
        
        function dX = backward(this, X, ~, dZ, ~)
            dX = nnet.internal.cnngpu.convolveBackwardData2D( ...
                X, this.Weights.Value, dZ, 0, 0, 1, 1);
        end
        
        function gradients = gradients(this, X, dZ)
            % gradients    Compute the gradients of the loss w.r.t. the
            % learnable parameters
            gradients{1} = nnet.internal.cnngpu.convolveBackwardFilter2D( ...
                X, this.Weights.Value, dZ, 0, 0, 1, 1);
            gradients{2} = nnet.internal.cnngpu.convolveBackwardBias2D(dZ);
        end
        
        function this = inferSize(this, inputSize)
            if ~this.HasSizeDetermined
                this.InputSize = inputSize;
                this = this.matchWeightsToLayerSize();
            end
        end
        
        function outputSize = forwardPropagateSize(this, inputSize)
            if ~this.HasSizeDetermined
                error('nnet_cnn:internal:cnn:layer:FullyConnected:ForwardPropagateSizeWithNoInputSize', ...
                    'An input size for the layer must be defined in order to call forwardPropagateSize.')
            else
                filterSize = this.InputSize(1:2);
                outputHeightAndWidth = floor(inputSize(1:2) - filterSize) + 1;
                outputSize = [ outputHeightAndWidth this.NumNeurons ];
            end
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = ( ~this.HasSizeDetermined || isequal(this.InputSize, inputSize) );
        end
        
        function this = initializeLearnableParameters(this, precision)
            if isempty(this.Weights.Value)
                % Initialize only if it is empty
                weightsSize = [this.InputSize this.NumNeurons];
                this.Weights.Value = iInitializeGaussian( weightsSize, precision );
            else
                % Cast to desired precision
                this.Weights.Value = precision.cast(this.Weights.Value);
            end
            
            if isempty(this.Bias.Value)
                % Initialize only if it is empty
                biasSize = [1 1 this.NumNeurons];
                this.Bias.Value = iInitializeConstant( biasSize, precision );         
            else
                % Cast to desired precision
                this.Bias.Value = precision.cast(this.Bias.Value);
            end
        end
        
        function this = prepareForTraining(this)
            % prepareForTraining   Prepare this layer for training
            %   Before this layer can be used for training, we need to
            %   convert the learnable parameters to use the class
            %   TrainingLearnableParameter.
            
            numParameters = numel(this.LearnableParameters);
            newLearnableParameters = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter.empty();
            for i = 1:numParameters
                newLearnableParameters(i) = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter();
                newLearnableParameters(i).Value = gpuArray(this.LearnableParameters(i).Value);
                newLearnableParameters(i).LearnRateFactor = this.LearnableParameters(i).LearnRateFactor;
                newLearnableParameters(i).L2Factor = this.LearnableParameters(i).L2Factor;
            end
            this.LearnableParameters = newLearnableParameters;
        end
        
        function this = prepareForPrediction(this)
            % prepareForPrediction   Prepare this layer for prediction
            %   Before this layer can be used for prediction, we need to
            %   convert the learnable parameters to use the class
            %   PredictionLearnableParameter.
            
            numParameters = numel(this.LearnableParameters);
            newLearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
            for i = 1:numParameters
                newLearnableParameters(i) = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
                newLearnableParameters(i).Value = gather(this.LearnableParameters(i).Value);
                newLearnableParameters(i).UseGPU = false;
                newLearnableParameters(i).LearnRateFactor = this.LearnableParameters(i).LearnRateFactor;
                newLearnableParameters(i).L2Factor = this.LearnableParameters(i).L2Factor;
            end
            this.LearnableParameters = newLearnableParameters;
        end
        
        function this = setupForHostPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.FullyConnectedHostStrategy();
            this.LearnableParameters(1).UseGPU = false;
            this.LearnableParameters(2).UseGPU = false;
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.FullyConnectedGPUStrategy();
            this.LearnableParameters(1).UseGPU = true;
            this.LearnableParameters(2).UseGPU = true;
        end
        
        function weights = get.Weights(this)
            weights = this.LearnableParameters(this.WeightsIndex);
        end
        
        function this = set.Weights(this, weights)
            this.LearnableParameters(this.WeightsIndex) = weights;
            if this.HasSizeDetermined
                this = this.matchWeightsToLayerSize();
            end
        end
        
        function bias = get.Bias(this)
            bias = this.LearnableParameters(this.BiasIndex);
        end
        
        function this = set.Bias(this, bias)
            this.LearnableParameters(this.BiasIndex) = bias;
            this = this.matchBiasToLayerSize();
        end
        
        function tf = get.HasSizeDetermined(this)
            tf = ~isempty( this.InputSize );
        end
    end
    
    methods (Access = private)
        function this = matchWeightsToLayerSize(this)
            % matchWeightsToLayerSize    Reshape weights from a matrix into
            % a 4-D array.
            requiredSize = [this.InputSize this.NumNeurons];
            weights = this.Weights.Value;
            if isequal( size( weights ), requiredSize )
                % Weights are the right size -- nothing to do
            elseif isempty( weights )
                % Weights are empty -- nothing we can do
            elseif ismatrix( weights )
                % Weights are 2d -- need to transpose and reshape to 4d
                % Transpose is needed since the user would set
                % it as [output input] instead of [input output].
                this.Weights.Value = reshape( weights', requiredSize );
            else
                % There are three possibilities and this is a fourth state
                % therefore somehting has gone wrong
                warning( 'nnet:internal:cnn:FullyConnected:InvalidState', 'Invalid state' );
            end
        end
        
        function this = matchBiasToLayerSize(this)
            % matchBiasToLayerSize   Reshape biases from a matrix into a
            % 3-D array.
            requiredSize = [1 1 this.NumNeurons];
            bias = this.Bias.Value;            
            if isequal( size( bias ), requiredSize )
                % Biases are the right size -- nothing to do
            elseif isempty( bias )
                % Biases are empty -- nothing we can do
            elseif ismatrix( bias )
                % Biases are 2d -- need to reshape to 3d
                this.Bias.Value = reshape(bias, requiredSize);
            end
        end
    end
end

function parameter = iInitializeGaussian(parameterSize, precision)
parameter = gpuArray( ...
    precision.cast( ...
    iNormRnd(0, 0.01, parameterSize) ) );
end

function parameter = iInitializeConstant(parameterSize, precision)
parameter = gpuArray( ...
    precision.cast( ...
    zeros(parameterSize) ) );
end

function out = iNormRnd(mu, sigma, outputSize)
    % iNormRnd  Returns an array of size 'outputSize' chosen from a
    % normal distribution with mean 'mu' and standard deviation 'sigma'
    out = randn(outputSize) .* sigma + mu;
end