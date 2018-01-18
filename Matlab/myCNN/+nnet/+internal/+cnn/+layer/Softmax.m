classdef Softmax < nnet.internal.cnn.layer.Layer
    % Softmax   Implementation of the softmax layer
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();

        % Name (char array)   A name for the layer
        Name
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'softmax'
    end
            
    properties(SetAccess = private)
        % HasSizeDetermined   Specifies if all size parameters are set
        %   For a softmax layer, this is always true.
        HasSizeDetermined = true
    end
    
    properties(Access = private)
        % ExecutionStrategy   The execution strategy for this layer
        %   This object 
        ExecutionStrategy
    end
    
    methods
        function this = Softmax(name)
            this.Name = name;
            
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.SoftmaxGPUStrategy();
        end
        
        function [Z, memory] = forward(this, X)
            [Z, memory] = this.ExecutionStrategy.forward(X);
        end
        
        function dX = backward(~, ~, Z, dZ, ~)
            dX = nnet.internal.cnngpu.softmaxBackward2D(Z, dZ);
        end
        
        function gradients = gradients( ~, ~, ~ )
            % gradients    No-op since this layer does not contain any
            % learnable parameters
            gradients = [];
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            outputSize = inputSize;
        end
        
        function this = inferSize(this, ~)
        end
        
        function tf = isValidInputSize(~, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            
            % A valid input size has 2 or 3 dimensions, with the first two
            % dimensions representing a vector
            tf = iIsVectorSize(inputSize(1:2)) && numel(inputSize)<=3;
        end
        
        function this = initializeLearnableParameters(this, ~)
        end
        
        function this = prepareForTraining(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter.empty();
        end
        
        function this = prepareForPrediction(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        end
        
        function this = setupForHostPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.SoftmaxHostStrategy();
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.SoftmaxGPUStrategy();
        end
    end
end

function tf = iIsVectorSize(inputSize)
tf = inputSize(1)==1 || inputSize(2)==1;
end