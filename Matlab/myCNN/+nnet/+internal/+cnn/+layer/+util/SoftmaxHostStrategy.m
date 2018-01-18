classdef SoftmaxHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % SoftmaxHostStrategy   Execution strategy for running the softmax layer on the host

    %   Copyright 2016 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X)
            Z = nnet.internal.cnnhost.softmaxForward2D(X);
            memory = [];
        end
    end
end