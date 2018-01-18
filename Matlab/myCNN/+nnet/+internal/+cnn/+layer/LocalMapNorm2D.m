classdef LocalMapNorm2D < nnet.internal.cnn.layer.Layer
    % LocalMapNorm2D   Implementation of the local response normalization layer (channel-wise)
    %
    %   A local response normalization layer that performs channel-wise
    %   normalization. For each element in the input x, we compute a
    %   normalized value y using the following formula:
    %
    %       y = x/(K + Alpha*sum(x(:,:,startChannel:stopChannel).^2, 3)/windowChannelSize)^Beta
    %
    %   Where:
    %       
    %       startChannel = max(channelIndex - floor((windowChannelSize - 1)/2), 1);
    %       stopChannel = min(channelIndex + ceil((windowChannelSize - 1)/2), numChannels);
    %       channelIndex is the index of the channel for the element x.
    %       numChannels is the number of channels for the input.
    %
    %   This can be seen as a form of lateral inhibition between channels.
    
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
        DefaultName = 'crossnorm'
    end
            
    properties(SetAccess = private)
        % HasSizeDetermined   Specifies if all size parameters are set
        %   For this layer, there are no size parameters to set.
        HasSizeDetermined = true
        
        % WindowChannelSize   Size of the window for normalization
        %   The size of a window which controls the number of channels that are
        %   used for the normalization of each element. For example, if
        %   this value is 3, each element will be normalized by its
        %   neighbours in the previous channel and the next channel. If
        %   WindowChannelSize is even, then the window will be asymmetric. For
        %   example, if it is 4, each element is normalized by its
        %   neighbour in the previous channel, and by its neighbours in the
        %   next two channels.  The value must be a positive integer.
        WindowChannelSize
        
        % Alpha   Multiplier for the normalization term
        %   The Alpha term in the normalization formula.
        Alpha
        
        % Beta   Exponent for the normalization term
        %   The Beta term in the normalization formula.
        Beta
        
        % K   Additive constant for the normalization term
        %   The K term in the normalization formula.
        K
    end
    
    properties(Access = private)
        ExecutionStrategy
    end
    
    methods
        function this = LocalMapNorm2D(name, windowChannelSize, alpha, beta, k)
            
            this.Name = name;
            
            % Set hyperparameters
            this.WindowChannelSize = windowChannelSize;
            this.Alpha = alpha;
            this.Beta = beta;
            this.K = k;
            
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.LocalMapNorm2DGPUStrategy();
        end
        
        function [Z, memory] = forward(this, X)
            [Z, memory] = this.ExecutionStrategy.forward( ...
                X, this.WindowChannelSize, this.Alpha, this.Beta, this.K);
        end
        
        function dX = backward(this, X, Z, dZ, ~)
            dX = nnet.internal.cnngpu.localMapNormBackward2D( ...
                Z, dZ, X, this.WindowChannelSize, this.Alpha, this.Beta, this.K);
        end
        
        function gradients = gradients( ~, ~, ~ )
            % gradients    No-op since this layer does not contain any
            % learnable parameters
            gradients = [];
        end
        
        function this = inferSize(this, ~)
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            outputSize = inputSize;
        end
        
        function tf = isValidInputSize(~, ~)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = true;
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
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.LocalMapNorm2DHostStrategy();
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.LocalMapNorm2DGPUStrategy();
        end
    end
end