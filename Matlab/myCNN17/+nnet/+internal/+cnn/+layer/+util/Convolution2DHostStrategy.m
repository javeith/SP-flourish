classdef Convolution2DHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % Convolution2DHostStrategy   Execution strategy for running the convolution on the host
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, ...
                weights, bias, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride)
            Z = nnet.internal.cnnhost.convolveForward2D( ...
                X, weights, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride);
            Z = Z + bias;
            memory = [];
        end
        
        function [dX,dW] = backward( ~, ...
                X, weights, dZ, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                strideHeight, strideWidth)
            dX = nnet.internal.cnnhost.convolveBackwardData2D( ...
                X, weights, dZ, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                strideHeight, strideWidth);
            
            dW{1} = nnet.internal.cnnhost.convolveBackwardFilter2D( ...
                X, weights, dZ, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                strideHeight, strideWidth);
            dW{2} = nnet.internal.cnnhost.convolveBackwardBias2D(dZ);
        end
        
    end
end