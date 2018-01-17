classdef ImageInput < nnet.internal.cnn.layer.Layer
    % ImageInput   Image input layer
    
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
        DefaultName = 'imageinput'
    end
        
    properties (SetAccess = private)        
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined
        
        % InputSize (vector of int)     Size of the input [height x width x
        %                               channels ]
        InputSize                      
        
        % Transforms      Vector of ImageTransform objects.
        Transforms
        
        % TrainTransforms Vector of ImageTransform objects.
        TrainTransforms                    
    end       
    
    properties(Dependent)
        % AverageImage The average training image. Stored when the user
        % requests zero centering.
        AverageImage
    end
    
    methods
        function this = ImageInput(name, inputSize, normalization, augmentations)
            % Input  Constructor for the layer
            this.Name = name;
            this.InputSize = inputSize;
            this.Transforms = normalization;
            this.TrainTransforms = augmentations;
            this.HasSizeDetermined = true;            
        end
        
        function [Z, memory] = forward( ~, X )
            % forward   Forward input data through the layer and output the result
            Z = X;
            memory = [];
        end
        
        function dX = backward( ~, ~, ~, ~, ~ )
            % backward  Return empty value
            dX = [];
        end
        
        function gradients = gradients( ~, ~, ~ ) %#ok<STOUT>
            % gradients    Throw an error since it should not be possible
            % to call gradients on this layer
            error('nnet:internal:cnn:layer:ImageInput:GradientsProhibited', ...
                'Error: the function gradients cannot be called on ImageInput layers');
        end
        
        function outputSize = forwardPropagateSize(this, ~)
            % forwardPropagateSize  Output the size of the layer 
            outputSize = this.InputSize;
        end     
        
        function this = inferSize(this, ~)
            % inferSize     no-op since this layer has nothing that can be
            %               inferred
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = isequal( inputSize, this.InputSize );
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
        
        function tf = isValidTrainingImageSize(this, trainingImageSize)
            % isValidTrainingImageSize   True if the training image size is
            % valid for this network
            
            imageSizeAfterTransforms = trainingImageSize;
            for i = 1:numel(this.TrainTransforms)
                imageSizeAfterTransforms = this.TrainTransforms(i).forwardPropagateSize(imageSizeAfterTransforms);
            end
            tf = isequal(this.InputSize, imageSizeAfterTransforms);
        end
        
        function I = get.AverageImage(this)
            % Get average image from internal transform layer. 
            if ~isempty(this.Transforms)
                I = this.Transforms.AverageImage;
            else
                I = [];
            end
        end               
        
        function this = set.AverageImage(this, val)
            % Update value of AverageImage. Forwards to transform
            % set.AverageImage method.
            if ~isempty(this.Transforms)
                % Only the zero center transforms is available.
                [h, w, c] = size(val); % Getting size this way allows to deal with a single grayscale image
                assert(isequal([h w c],this.InputSize), ...
                    'Average image should be same size as input image');                
                this.Transforms.AverageImage = val;
            end            
        end
    end       
end

