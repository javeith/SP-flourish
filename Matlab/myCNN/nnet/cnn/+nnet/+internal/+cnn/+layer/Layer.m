classdef (Abstract) Layer
    % Layer    Interface for convolutional neural network layers
    
    %   Copyright 2015-2016 The MathWorks, Inc.    
    
    properties (Abstract)
        % LearnableParameters   Learnable parameters for the layer
        % (Vector of nnet.internal.cnn.layer.LearnableParameter)
        LearnableParameters

        % Name (char array)
        Name
    end
    
    properties (Abstract, SetAccess = private)
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined
    end
    
    properties (Abstract, Constant)
        % DefaultName   Default layer's name. This will be assigned in case
        % the user leaves an empty name.
        DefaultName
    end
         
    methods (Abstract)
        % forward   Forward input data through the layer and output the result
        %
        % Syntax
        %   [Z, memory] = forward( aLayer, X )
        %
        % Inputs
        %   aLayer - the layer to forward thru
        %   X - the input to forward propagate thru the layer
        %
        % Outputs
        %   Z - the output of forward propagation thru the layer
        %   memory - "memory" produced by forward propagation thru the layer        
        [Z, memory] = forward( aLayer, X )
        
        % backward    Back propagate the derivative of the loss function
        % thru one layer
        %
        % For notation, assume the that a layer is represented by the
        % function Z = f(X) and derivatives of the loss function with
        % respect to X and Z are dX and dZ.
        %
        % Syntax
        %   dX = backward( aLayer, X, Z, dZ, memory )
        %
        % Inputs
        %   aLayer - the layer to backprop thru
        %   X - the input that was used for forward propagation thru the
        %       layer
        %   Z - the output from forward propagation thru the layer
        %   dZ - the derivative of the loss function with respect to Z.
        %       This is usually obtained via back-propagation from the next
        %       layer in the network.
        %   memory - whatever "memory" that was produced by forward
        %       propagation thru the layer
        %
        % Outputs
        %   dX - the derivative of the loss function with respect to X
        %
        % See also: forward
        dX = backward( aLayer, X, Z, dZ, memory )

        % gradients   Returns the gradients of the loss with respect to the
        % learnable parameters for the layer.
        %
        % Syntax
        %   gradients = gradients( aLayer, X, dZ )
        %
        % Inputs
        %   aLayer - the layer to compute learnable parameters gradients of
        %   X - the input that was used for forward propagation thru the
        %       layer
        %   dZ - the derivative of the loss function with respect to Z.
        %       This is usually obtained via back-propagation from the next
        %       layer in the network.
        %
        % Outputs
        %   gradients - cell array of gradients with one element for each
        %               learnable parameter
        gradients = gradients( aLayer, X, dZ )
                    
        % forwardPropagateSize    The size of the output from the layer for
        % a given size of input
        outputSize = forwardPropagateSize(this, inputSize)   

        % inferSize    Infer the size of the learnable parameters based
        % on the input size
        this = inferSize(this, inputSize)
        
        % isValidInputSize   Check if the layer can accept an input of a
        % certain size
        tf = isValidInputSize(inputSize)
    
        % initializeLearnableParameters    Initialize learnable parameters
        % using their initializer
        this = initializeLearnableParameters(this, precision)
        
        % prepareForTraining   Prepare the layer for training
        this = prepareForTraining(this)
        
        % prepareForPrediction   Prepare the layer for prediction
        this = prepareForPrediction(this)
        
        % setupForHost   Prepare this layer for host prediction
        this = setupForHostPrediction(this)
        
        % setupForGPU   Prepare this layer for GPU prediction
        this = setupForGPUPrediction(this)
    end        

    methods(Hidden, Static)
        function [layerIdx, layerNames] = findLayerByName(layers, name)
            % findLayerByName   Find index of layer by name
            %
            % Returns an index to layer that with a matching name. An empty is
            % returned if a layer with that name is not found. Multiple matches
            % may be returned. Callers must decide how to handle these cases.
            layerNames = nnet.internal.cnn.layer.Layer.getLayerNames(layers);
            layerIdx = find(strcmp(name, layerNames));          
        end
        
        function layerNames = getLayerNames(layers)
            layerNames = cellfun(@(x)x.Name, layers, 'UniformOutput', false);
        end
    end
end