classdef FullyConnectedHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % FullyConnectedHostStrategy   Execution strategy for running the fully connected layer on the host

    %   Copyright 2016 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, weights, bias)
            Z = nnet.internal.cnnhost.convolveForward2D( ...
                X, weights, 0, 0, 1, 1);
            Z = Z + bias;
            memory = [];
        end
    end
end