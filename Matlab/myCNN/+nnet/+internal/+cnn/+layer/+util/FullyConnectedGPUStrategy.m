classdef FullyConnectedGPUStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % FullyConnectedGPUStrategy   Execution strategy for running the fully connected layer on the GPU

    %   Copyright 2016 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, weights, bias)
            Z = nnet.internal.cnngpu.convolveForward2D(...
                X, weights, 0, 0, 1, 1);
            Z = arrayfun(@plus, Z, bias);
            memory = [];
        end
    end
end