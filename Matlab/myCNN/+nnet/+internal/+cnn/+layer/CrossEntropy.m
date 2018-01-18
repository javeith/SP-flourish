classdef CrossEntropy < nnet.internal.cnn.layer.OutputLayer
    % CrossEntropy   Cross entropy loss output layer
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % Name (char array)   A name for the layer
        Name
        
        % ClassNames   The names of the classes
        ClassNames
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'classoutput'
    end
    
    properties (SetAccess = private)
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined
        
        % NumClasses (scalar int)   Number of classes
        NumClasses
    end
    
    methods
        function this = CrossEntropy(name, numClasses)
            % Output  Constructor for the layer
            % creates an output layer with the following parameters:
            %
            %   name                - Name for the layer
            %   numClasses          - Number of classes. [] if it has to be
            %                       determined later
            this.Name = name;
            if(isempty(numClasses))
                this.NumClasses = [];
            else
                this.NumClasses = numClasses;
            end
            this.ClassNames = {};
            this.HasSizeDetermined = ~isempty( numClasses );
        end
        
        function [Z, memory] = forward( ~, X )
            % forward   Forward input data through the layer and output the
            % result
            Z = X;
            memory = [];
        end
        
        function dX = backward( ~, ~, ~, ~, ~ ) %#ok<STOUT>
            % backward    Throw an error since it should not be possible to
            % call backward on this layer
            error('nnet:internal:cnn:layer:CrossEntropy:BackwardProhibited', ...
                'Error: the function backward cannot be called on CrossEntropy layers');
        end
        
        function gradients = gradients( ~, ~, ~ ) %#ok<STOUT>
            % gradients    Throw an error since it should not be possible
            % to call gradients on this layer
            error('nnet:internal:cnn:layer:CrossEntropy:GradientsProhibited', ...
                'Error: the function gradients cannot be called on CrossEntropy layers');
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            % forwardPropagateSize  Output the size of the layer based on
            % the input size
            outputSize = inputSize;
        end
        
        function this = inferSize(this, inputSize)
            % inferSize    Infer the number of classes based on the input
            this.NumClasses = inputSize(3);
            this.HasSizeDetermined = true;
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size.
            tf = numel(inputSize)==3 && isequal(inputSize, [1 1 this.NumClasses]);
        end
        
        function this = initializeLearnableParameters(this, ~)
            % initializeLearnableParameters     no-op since there are no
            % learnable parameters
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
        
        function loss = forwardLoss( ~, Y, T )
            % forwardLoss    Return the cross entropy loss between estimate
            % and true responses averaged by the number of observations
            %
            % Syntax:
            %   loss = layer.forwardLoss( Y, T );
            %
            % Inputs:
            %   Y   Predictions made by network, 1-by-1-by-numClasses-by-numObs
            %   T   Targets (actual values), 1-by-1-by-numClasses-by-numObs            
            numObservations = size(Y, 4);
            loss = -1/numObservations * sum(sum(T.*log(iBoundAwayFromZero(Y))));
        end
        
        function dX = backwardLoss( ~, Y, T )
            % backwardLoss    Back propagate the derivative of the loss
            % function
            %
            % Syntax:
            %   dX = layer.backwardLoss( Y, T );
            %
            % Inputs:
            %   Y   Predictions made by network, 1-by-1-by-numClasses-by-numObs
            %   T   Targets (actual values), 1-by-1-by-numClasses-by-numObs
            dX = -T./iBoundAwayFromZero(Y);
        end
    end
end

function xBounded = iBoundAwayFromZero(x)
xBounded = x;
precision = class(gather(x));
xBounded = iBoundSmallPositiveValuesAndZero(xBounded, precision);
xBounded = iBoundSmallNegativeValues(xBounded, precision);
end

function xBounded = iBoundSmallPositiveValuesAndZero(xBounded, precision)
xBounded((0 <= xBounded) & (xBounded < eps(precision))) = eps(precision);
end

function xBounded = iBoundSmallNegativeValues(xBounded, precision)
xBounded((-eps(precision) < xBounded) & (xBounded < 0)) = -eps(precision);
end