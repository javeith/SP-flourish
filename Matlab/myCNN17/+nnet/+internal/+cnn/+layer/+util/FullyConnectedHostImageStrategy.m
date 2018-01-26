classdef FullyConnectedHostImageStrategy < nnet.internal.cnn.layer.util.FullyConnectedImageStrategy
    % FullyConnectedHostImageStrategy   Execution strategy for running the
    % fully connected layer on the host with image inputs

    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function X = sendToDevice(~, X)
            % No operation required for the host
        end
        
        function Z = convolveForward(~, X, weights)
            Z = nnet.internal.cnnhost.convolveForward2D(X, weights, 0, 0, 0, 0, 1, 1);
        end
        
        function dX = convolveBackwardData(~, X, weights, dZ)
            dX = nnet.internal.cnnhost.convolveBackwardData2D(X, weights, dZ, 0, 0, 0, 0, 1, 1);
        end
        
        function dW = convolveBackwardFilter(~, X, weights, dZ)
            dW = nnet.internal.cnnhost.convolveBackwardFilter2D(X, weights, dZ, 0, 0, 0, 0, 1, 1);
        end
        
        function dB = convolveBackwardBias(~, dZ)
            dB = nnet.internal.cnnhost.convolveBackwardBias2D(dZ);
        end
    end
end