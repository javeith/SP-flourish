classdef AveragePooling2D < nnet.internal.cnn.layer.Layer
    % AveragePooling2D   Average 2D pooling layer implementation
    
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
        DefaultName = 'avgpool'
    end
        
    properties (SetAccess = private)
        % HasSizeDetermined   Specifies if all size parameters are set
        %   For pooling layers, this is always true.
        HasSizeDetermined = true

        % PoolSize   The height and width of a pooling region
        %   The size the pooling regions. This is a vector [h w] where h is
        %   the height of a pooling region, and w is the width of a pooling
        %   region.
        PoolSize
        
        % Stride   The vertical and horizontal stride
        %   The step size for traversing the input vertically and
        %   horizontally. This is a vector [u v] where u is the vertical
        %   stride and v is the horizontal stride.
        Stride
        
        % Padding   The vertical and horizontal padding
        %   The padding that is applied to the input for this layer. This
        %   is a vector [a b] where a is the padding applied to the top and
        %   bottom of the input, and b is the padding applied to the left
        %   and right.
        Padding
    end
    
    properties(Access = private)
        ExecutionStrategy
    end
    
    methods
        function this = AveragePooling2D(name, poolSize, stride, padding)
            this.Name = name;
            
            % Set hyperparameters
            this.PoolSize = poolSize;
            this.Stride = stride;
            this.Padding = padding;
            
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.AveragePooling2DGPUStrategy();
        end
        
        function [Z, memory] = forward(this, X)
            [Z, memory] = this.ExecutionStrategy.forward(...
                X, ...
                this.PoolSize(1), this.PoolSize(2), ...
                this.Padding(1), this.Padding(2), ...
                this.Stride(1), this.Stride(2));
        end
        
        function dX = backward(this, X, Z, dZ, ~)
            dX = nnet.internal.cnngpu.poolingAverageBackward2D(...
                Z, dZ, X, ...
                this.PoolSize(1), this.PoolSize(2), ...
                this.Padding(1), this.Padding(2), ...
                this.Stride(1), this.Stride(2));
        end
        
        function gradients = gradients( ~, ~, ~ )
            % gradients    No-op since this layer does not contain any
            % learnable parameters
            gradients = [];
        end

        function outputSize = forwardPropagateSize(this, inputSize)
            outputHeightAndWidth = floor((inputSize(1:2) + 2*this.Padding - this.PoolSize)./this.Stride) + 1;
            outputMaps = inputSize(3);
            outputSize = [outputHeightAndWidth outputMaps];
        end

        function this = inferSize(this, ~)
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = all( this.PoolSize <= inputSize(1:2) + 2*this.Padding );
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
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.AveragePooling2DHostStrategy();
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.AveragePooling2DGPUStrategy();
        end
    end
end