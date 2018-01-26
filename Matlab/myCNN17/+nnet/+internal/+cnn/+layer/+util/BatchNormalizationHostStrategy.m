classdef BatchNormalizationHostStrategy
    % BatchNormalizationHostStrategy   Execution strategy for running batch normalization on the host
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forwardTrain(~, X, beta, gamma, epsilon)
            [Z,batchMean,batchInvVar] = ...
                nnet.internal.cnnhost.batchNormalizationForwardTrain(X, beta, gamma, epsilon);
            memory = {batchMean, batchInvVar};
        end
        
        function Z = forwardPredict(~, X, beta, gamma, epsilon, inputMean, inputVar)
            Z = nnet.internal.cnnhost.batchNormalizationForwardPredict(X, beta, gamma, epsilon, inputMean, inputVar);
        end
        
        function [dX,dW] = backward(~, ~, dZ, X, gamma, epsilon, memory)
            [batchMean, batchInvVar] = deal(memory{:});
            [dX,dBeta,dGamma] = ...
                nnet.internal.cnnhost.batchNormalizationBackward(dZ, X, gamma, epsilon, batchMean, batchInvVar);
            dW = {dBeta, dGamma};
        end
        
    end
end
