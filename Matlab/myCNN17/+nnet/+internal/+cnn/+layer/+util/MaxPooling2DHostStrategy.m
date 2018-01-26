classdef MaxPooling2DHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % MaxPooling2DHostStrategy   Execution strategy for running the max pooling on the host
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, ...
                poolHeight, poolWidth, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride)
            Z = nnet.internal.cnnhost.poolingMaxForward2D(X, ...
                poolHeight, poolWidth, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride);
            memory = [];
        end
        
        function [dX,dW] = backward(~, Z, dZ, X, ...
                poolHeight, poolWidth, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride)
            dX = nnet.internal.cnnhost.poolingMaxBackward2D(...
                Z, dZ, X, ...
                poolHeight, poolWidth, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride);
            dW = [];
        end
    end
end