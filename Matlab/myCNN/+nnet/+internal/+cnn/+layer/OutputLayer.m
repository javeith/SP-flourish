classdef OutputLayer < nnet.internal.cnn.layer.Layer
    % OutputLayer     Interface for convolutional neural network output layers
    
    %   Copyright 2015 The MathWorks, Inc.

    methods (Abstract)
        % forwardLoss    Return the loss between the output obtained from
        % the network and the expected output
        %
        % Inputs
        %   anOutputLayer - the output layer to forward the loss thru
        %   Z - the output from forward propagation thru the layer
        %   T - the expected output
        %
        % Outputs
        %   loss - the loss between Z and T
        loss = forwardLoss( anOutputLayer, Z, T)
        
        % backwardLoss    Back propagate the derivative of the loss function
        %
        % Inputs
        %   anOutputLayer - the output layer to backprop the loss thru
        %   Z - the output from forward propagation thru the layer
        %   T - the expected output
        %
        % Outputs
        %   dX - the derivative of the loss function with respect to X        
        dX = backwardLoss( anOutputLayer, Z, T)
    end
end
