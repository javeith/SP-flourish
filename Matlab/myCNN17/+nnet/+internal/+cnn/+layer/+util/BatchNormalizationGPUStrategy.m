classdef BatchNormalizationGPUStrategy
    % BatchNormalizationGPUStrategy   Execution strategy for running batch normalization on the GPU
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forwardTrain(~, X, beta, gamma, epsilon)
            [Z,batchMean,batchInvVar] = ...
                nnet.internal.cnngpu.batchNormalizationForwardTrain(X, beta, gamma, epsilon);
            memory = {batchMean, batchInvVar};
        end
        
        function Z = forwardPredict(~, X, beta, gamma, epsilon, inputMean, inputVar)
            % Check for large batch sizes. A bug in cuDNN means it cannot
            % predict more than 1024 at once.
            batchMax = 1024;
            batchSize = size(X,4);
            if batchSize<=batchMax
                % Safe to predict the entire batch in one go
                Z = nnet.internal.cnngpu.batchNormalizationForwardPredict(X, ...
                    beta, gamma, epsilon, inputMean, inputVar);
                
            else
                % Pre-allocate Z
                Z = gpuArray(zeros(size(X),'like',X));
                
                % Predict on sub-batches of X to avoid the cuDNN bug
                numSubBatches = ceil(batchSize/batchMax);
                for ii=1:numSubBatches
                    batchStart = (ii-1)*batchMax + 1;
                    batchEnd = min(batchSize, ii*batchMax);
                    
                    % Run the sub-batch
                    Z(:,:,:,batchStart:batchEnd) = nnet.internal.cnngpu.batchNormalizationForwardPredict(X(:,:,:,batchStart:batchEnd), ...
                        beta, gamma, epsilon, inputMean, inputVar);
                end
                
            end
        end
        
        function [dX,dW] = backward(~, ~, dZ, X, gamma, epsilon, memory)
            [batchMean, batchInvVar] = deal(memory{:});
            [dX,dBeta,dGamma] = ...
                nnet.internal.cnngpu.batchNormalizationBackward(dZ, X, gamma, epsilon, batchMean, batchInvVar);
            dW = {dBeta, dGamma};
        end
        
    end
end
