classdef LSTMHostReturnLastStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % LSTMHostReturnLastStrategy   Execution strategy for running LSTM on
    % the host with ReturnSequence false.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function [Y, memory] = forward(~, X, W, R, b, c0, y0)
            [Y, C, G] = nnet.internal.cnnhost.lstmForwardReturnLast(X, W, R, b, c0, y0);
            memory.CellState = C;
            memory.Gates = G;
        end
        
        function [dX, dW, dR, dB] = backward(~, X, W, R, b, c0, y0, Y, memory, dZ)
            C = memory.CellState;
            G = memory.Gates;
            [dX, dW, dR, dB] = nnet.internal.cnnhost.lstmBackwardReturnLast(X, W, R, b, c0, y0, Y, C, G, dZ);
        end
    end
end
