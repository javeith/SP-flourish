classdef SoftmaxGPUStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % SoftmaxGPUStrategy   Execution strategy for running the softmax layer on the GPU

    %   Copyright 2016 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X)
            Z = nnet.internal.cnngpu.softmaxForward2D(X);
            memory = [];
        end
    end
end