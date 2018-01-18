classdef ReLUHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % ReLUHostStrategy   Execution strategy for running ReLU on the host
    
    %   Copyright 2016 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X)
            Z = nnet.internal.cnnhost.reluForward(X);
            memory = [];
        end
    end
end