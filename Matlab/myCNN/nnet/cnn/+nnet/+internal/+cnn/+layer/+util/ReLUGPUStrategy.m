classdef ReLUGPUStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % ReLUGPUStrategy   Execution strategy for running ReLU on the GPU
    
    %   Copyright 2016 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X)
            Z = nnet.internal.cnngpu.reluForward(X);
            memory = [];
        end
    end
end