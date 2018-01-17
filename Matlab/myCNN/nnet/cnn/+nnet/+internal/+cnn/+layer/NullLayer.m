classdef NullLayer < nnet.internal.cnn.layer.Layer
    % NullLayer     A layer that does nothing else than forwarding its
    % inputs to its outputs
    
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
        DefaultName = 'null'
    end
                  
    properties (SetAccess = private)
        % HasSizeDetermined   Always true for a null layer
        HasSizeDetermined = true;
    end
    
    methods
        function this = NullLayer(name)
            % NullLayer  Constructor for the layer
            this.Name = name;
        end
        
        function [Z, memory] = forward( ~, X )
            % forward   Forward input data through the layer and output the result
            Z = X;
            memory = [];
        end
        
        function dX = backward( ~, ~, ~, dZ, ~ )
            % backward    Back propagate the derivative of the loss function
            % thru the layer
            dX = dZ;
        end
        
        function gradients = gradients( ~, ~, ~ )
            % gradients    No-op since this layer does not contain any
            % learnable parameters
            gradients = [];
        end      
        
        function outputSize = forwardPropagateSize(~, inputSize)
            % forwardPropagateSize  Output the size of the layer based on
            %                       the input size
            outputSize = inputSize;
        end
        
        function this = inferSize(this, ~)
            % inferSize     no-op since this layer has nothing that can be
            %               inferred
        end
        
        function tf = isValidInputSize(~, ~)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = true;
        end
        
        function this = initializeLearnableParameters(this, ~)
            % initializeLearnableParameters     no-op since there are no
            %                                   learnable parameters
        end
        
        function this = prepareForTraining(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter.empty();
        end
        
        function this = prepareForPrediction(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        end
        
        function this = setupForHostPrediction(this)
        end
        
        function this = setupForGPUPrediction(this)
        end
    end
end
