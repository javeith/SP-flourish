classdef MeanSquaredError < nnet.internal.cnn.layer.RegressionLayer
    % MeanSquaredError   MeanSquaredError loss output layer
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % Name (char array)   A name for the layer
        Name
        
        % ResponseNames (cellstr)   The names of the responses
        ResponseNames
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'regressionoutput'
    end
    
    properties (SetAccess = private)
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined = true
    end
    
    methods
        function this = MeanSquaredError(name)
            % MeanSquaredError   Constructor for the layer
            this.Name = name;
            this.ResponseNames = {};
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            % forwardPropagateSize  Output the size of the layer based on
            % the input size
            outputSize = inputSize;
        end
        
        function this = inferSize(this, ~)
            
            % no-op since this layer has nothing that can be inferred
        end
        
        function tf = isValidInputSize(~, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size.
            tf = numel(inputSize)==3;
        end
        
        function this = initializeLearnableParameters(this, ~)
            
            % no-op since there are no learnable parameters
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
        
        function this = setupForHostTraining(this)
        end
        
        function this = setupForGPUTraining(this)
        end
        
        function loss = forwardLoss( ~, Y, T )
            % forwardLoss    Return the MSE loss between estimate
            % and true responses averaged by the number of observations
            %
            % Syntax:
            %   loss = layer.forwardLoss( Y, T );
            %
            % Inputs:
            %   Y   Predictions made by network, of size:
            %   height-by-width-by-numResponses-by-numObservations
            %   T   Targets (actual values), of size:
            %   height-by-width-by-numResponses-by-numObservations
            
            squares = 0.5*(Y-T).^2;
            numObservations = size( squares, 4 );
            loss = sum( squares (:) ) / numObservations;
        end
        
        function dX = backwardLoss( ~, Y, T )
            % backwardLoss    Back propagate the derivative of the loss
            % function
            %
            % Syntax:
            %   dX = layer.backwardLoss( Y, T );
            %
            % Inputs:
            %   Y   Predictions made by network, of size:
            %   height-by-width-by-numResponses-by-numObservations
            %   T   Targets (actual values), of size:
            %   height-by-width-by-numResponses-by-numObservations
            numObservations = size( Y, 4 );
            dX = (Y - T)./numObservations;
        end
    end
end
