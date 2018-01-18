classdef Convolution2D < nnet.internal.cnn.layer.Layer
    % Convolution2D   Implementation of the 2D convolution layer
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        % (Vector of nnet.internal.cnn.layer.learnable.PredictionLearnableParameter)
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();

        % Name (char array)   A name for the layer
        Name
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'conv'
    end
    
    properties (SetAccess = private)
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined
                
        % Hyperparameters
        
        % FilterSize  (1x2 int vector)  Size of each filter expressed in
        % height x width
        FilterSize
        
        % NumChannels (int)   The number of channels that the input to the
        % layer will have. [] if it has to be inferred later
        NumChannels
        
        % NumFilters (int)  The number of filters in the layer
        NumFilters
        
        % Stride (vector of int) Stride for each dimension
        Stride
        
        % Padding (vector of int) Padding for each dimension
        Padding
    end
    
    properties(Access = private)
        ExecutionStrategy
    end
    
    properties (Dependent)
        % Learnable Parameters (nnet.internal.cnn.layer.LearnableParameter)
        Weights
        Bias
    end
    
    properties (Constant, Access = private)
        % WeightsIndex  Index of the Weights into the LearnableParameter
        %               vector
        WeightsIndex = 1;
        
        % BiasIndex     Index of the Bias into the LearnableParameter
        %               vector
        BiasIndex = 2;
    end
    
    methods
        function this = Convolution2D(name, filterSize, numChannels, ...
                numFilters, stride, padding)
            % Convolution2D   Constructor for a Convolution2D layer
            %
            %   Create a 2D convolutional layer with the following
            %   compulsory parameters:
            %
            %       name            - Name for the layer
            %       filterSize      - Size of the filters [height x width]
            %       numChannels     - The number of channels that the input
            %                       to the layer will have. [] if it has to
            %                       be determined later
            %       numFilters      - The number of filters in the layer
            %       stride          - A vector specifying the stride for
            %                       each dimension [height x width]
            %       padding         - A vector specifying the padding for
            %                       each dimension [height x width]
            
            this.Name = name;
            
            % Set Hyperparameters
            this.FilterSize = filterSize;
            this.NumChannels = numChannels;
            this.HasSizeDetermined = ~isempty( numChannels );
            this.NumFilters = numFilters;
            this.Stride = stride;
            this.Padding = padding;
            
            % Set weights and bias to be LearnableParameter
            this.Weights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.Convolution2DGPUStrategy();
        end
        
        function [Z, memory] = forward( this, X )
            % forward   Forward input data through the layer and output the result
            if(this.usingFilterGroups())
                [Z, memory] = this.forwardTwoFilterGroups( X );
            else
                [Z, memory] = this.forwardNormal( X );
            end
        end
        
        function dX = backward( this, X, ~, dZ, ~ )
            % backward    Back propagate the derivative of the loss function
            % thru the layer
            if(this.usingFilterGroups())
                dX = this.backwardTwoFilterGroups(X, [], dZ, []);
            else
                dX = this.backwardNormal(X, [], dZ, []);
            end
        end
        
        function gradients = gradients(this, X, dZ)
            % gradients   Compute the gradients of the loss w.r.t. the
            % learnable parameters
            if(this.usingFilterGroups())
                gradients = this.gradientsTwoFilterGroups(X, dZ);
            else
                gradients = this.gradientsNormal(X, dZ);
            end
        end
        
        function this = inferSize(this, inputSize)
            % inferSize     Infer the number of channels based on the input size
            numChannels = iNumChannelsFromInputSize(inputSize);
            if (this.usingFilterGroups())
                % For filter groups, the number of channels is half the
                % size
                numChannels = numChannels/2;
            end
            this.NumChannels = numChannels;
            
            this.HasSizeDetermined = true;
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = this.isFilterSizeSmallerThanImage( inputSize ) && this.numFilterChannelsMatchesNumImageChannels( inputSize );
        end
        
        function outputSize = forwardPropagateSize(this, inputSize)
            % forwardPropagateSize    Output the size of the layer based on
            % the input size
            outputHeightAndWidth = floor((inputSize(1:2) + 2*this.Padding - this.FilterSize)./this.Stride) + 1;
            if(this.HasSizeDetermined)
                outputSize = [outputHeightAndWidth sum(this.NumFilters)];
            else
                outputSize = [outputHeightAndWidth NaN];
            end
        end
        
        function this = initializeLearnableParameters(this, precision)
            % initializeLearnableParameters    Initialize learnable
            % parameters using their initializer
            
            if isempty(this.Weights.Value)
                % Initialize only if it is empty
                weightsSize = [this.FilterSize, this.NumChannels, sum(this.NumFilters)];
                this.LearnableParameters(this.WeightsIndex).Value = iInitializeGaussian( weightsSize, precision );
            else
                % Cast to desired precision
                this.Weights.Value = precision.cast(this.Weights.Value);
            end
            
            if isempty(this.Bias.Value)
                % Initialize only if it is empty
                biasSize = [1, 1, sum(this.NumFilters)];
                this.LearnableParameters(this.BiasIndex).Value = iInitializeConstant( biasSize, precision );
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
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.Convolution2DHostStrategy();
            this.LearnableParameters(1).UseGPU = false;
            this.LearnableParameters(2).UseGPU = false;
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.Convolution2DGPUStrategy();
            this.LearnableParameters(1).UseGPU = true;
            this.LearnableParameters(2).UseGPU = true;
        end
        
        % Setter and getter for Weights and Bias
        % These make easier to address into the vector of LearnableParameters
        % giving a name to each index of the vector
        function weights = get.Weights(this)
            weights = this.LearnableParameters(this.WeightsIndex);
        end
        
        function this = set.Weights(this, weights)
            this.LearnableParameters(this.WeightsIndex) = weights;
        end
        
        function bias = get.Bias(this)
            bias = this.LearnableParameters(this.BiasIndex);
        end
        
        function this = set.Bias(this, bias)
            this.LearnableParameters(this.BiasIndex) = bias;
        end
    end
    
    methods(Access = private)
        function tf = usingFilterGroups(this)
            tf = not(numel(this.NumFilters) == 1);
        end
        
        function [Z, memory] = forwardNormal( this, X )
            Z = this.doForward(X,this.Weights.Value,this.Bias.Value);
            memory = [];
        end

        function [Z, memory] = forwardTwoFilterGroups( this, X )
            % forwardFilterGroup   Forward propagation for two filter groups
            
            [X1,X2] = iSplitDataAlongThirdDimension(X, this.NumChannels);
            
            [weights1, weights2] = iSplitWeightsAlongFourthDimension(this.Weights.Value, this.NumFilters);
            
            [bias1, bias2] = iSplitBiasAlongThirdDimension(this.Bias.Value, this.NumFilters);
            
            % Do forward propagation in two parallel branches
            Z1 = this.doForward(X1,weights1,bias1);
            Z2 = this.doForward(X2,weights2,bias2);
            
            % Stack the results back together again
            Z = cat(3, Z1, Z2);
            memory = [];
        end
        
        function Z = doForward(this, X, weights, bias)            
            Z = this.ExecutionStrategy.forward(X, weights, bias, ...
                this.Padding(1), this.Padding(2), ...
                this.Stride(1), this.Stride(2) );
        end

        function dX = backwardNormal(this, X, ~, dZ, ~)
            dX = this.doBackward(X, this.Weights.Value, dZ);
        end
        
        function dX = backwardTwoFilterGroups(this, X, ~, dZ, ~)
            % backwardFilterGroup   Backpropagation for two filter groups
            
            [X1, X2] = iSplitDataAlongThirdDimension(X, this.NumChannels);
            
            [weights1, weights2] = iSplitWeightsAlongFourthDimension( ...
                this.Weights.Value, this.NumFilters);
            
            [dZ1, dZ2] = iSplitDerivativeAlongThirdDimension(dZ, this.NumFilters);
            
            % Do backward propagation in two parallel branches
            dX1 = this.doBackward(X1, weights1, dZ1);
            dX2 = this.doBackward(X2, weights2, dZ2);
            
            dX = cat(3, dX1, dX2);
        end
        
        function dX = doBackward(this, X, weights, dZ)
            dX = nnet.internal.cnngpu.convolveBackwardData2D( ...
                X, weights, dZ, ...
                this.Padding(1), this.Padding(2), ...
                this.Stride(1), this.Stride(2) ...
                );
        end
        
        function gradients = gradientsNormal(this, X, dZ)            
            gradients = this.doGradients(X, this.Weights.Value, dZ);
        end
        
        function gradients = gradientsTwoFilterGroups(this, X, dZ)
            % gradientsTwoFilterGroups   Compute gradients for two filter groups
            
            [X1, X2] = iSplitDataAlongThirdDimension(X, this.NumChannels);
            
            [weights1, weights2] = iSplitWeightsAlongFourthDimension( ...
                this.Weights.Value, this.NumFilters);
            
            [dZ1, dZ2] = iSplitDerivativeAlongThirdDimension(dZ, this.NumFilters);
            
            % Do backpropagation in two parallel branches
            gradients1 = this.doGradients(X1, weights1, dZ1);
            gradients2 = this.doGradients(X2, weights2, dZ2);
            
            weightsGradient = cat(4, gradients1{1}, gradients2{1});
            biasGradient = cat(3, gradients1{2}, gradients2{2});
            
            gradients{1} = weightsGradient;
            gradients{2} = biasGradient;
        end
        
        function gradients = doGradients(this, X, weights, dZ)            
            gradients{1} = nnet.internal.cnngpu.convolveBackwardFilter2D( ...
                X, weights, dZ, ...
                this.Padding(1), this.Padding(2), ...
                this.Stride(1), this.Stride(2) ...
                );
            gradients{2} = nnet.internal.cnngpu.convolveBackwardBias2D(dZ);
        end
        
        function tf = isFilterSizeSmallerThanImage( this, inputSize )
            % The size of the image is given by the first two dimensions of the input size
            imageSize = inputSize(1:2);
            
            % Need to take padding into account when comparing image size and filter size
            tf = all( this.FilterSize <= imageSize + 2*this.Padding );
        end
        
        function tf = numFilterChannelsMatchesNumImageChannels( this, inputSize )
            numImageChannels = inputSize(3);
            % The total number of channels for the filters must take into
            % account wether filter groups are used
            numGroups = numel(this.NumFilters);
            totalNumFilterChannels = numGroups*this.NumChannels;
            tf = ~this.HasSizeDetermined || totalNumFilterChannels == numImageChannels;
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
function [data1, data2] = iSplitDataAlongThirdDimension(data, numChannels)
data1 = data(:,:,1:numChannels,:);
data2 = data(:,:,numChannels + 1:2*numChannels,:);
end

function [weights1, weights2] = iSplitWeightsAlongFourthDimension(weights, numFilters)
weights1 = weights(:,:,:,1:numFilters(1));
weights2 = weights(:,:,:,numFilters(1)+1:numFilters(1)+numFilters(2));
end

function [bias1, bias2] = iSplitBiasAlongThirdDimension(bias, numFilters)
bias1 = bias(:,:,1:numFilters(1));
bias2 = bias(:,:,numFilters(1)+1:numFilters(1)+numFilters(2));
end

function [dZ1, dZ2] = iSplitDerivativeAlongThirdDimension(dZ, numFilters)
dZ1 = dZ(:,:,1:numFilters(1),:);
dZ2 = dZ(:,:,numFilters(1)+1:numFilters(1)+numFilters(2),:);
end

function numChannels = iNumChannelsFromInputSize(inputSize)
% iNumChannelsFromInputSize   The number of channels is the third element
% in inputSize. If inputSize doesn't have three elements, then it is
% implicitly 1.
if numel(inputSize)<3
    numChannels = 1;
else
    numChannels = inputSize(3);
end
end
