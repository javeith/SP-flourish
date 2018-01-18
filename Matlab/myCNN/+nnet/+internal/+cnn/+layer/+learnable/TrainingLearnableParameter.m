classdef TrainingLearnableParameter < nnet.internal.cnn.layer.learnable.LearnableParameter
    % TrainingLearnableParameter   Learnable parameter for use at training time
    %
    %   This class is used to represent learnable parameters during
    %   training. The representation that is used is very simple. the
    %   learnable parameter is stored in the property Value, which can be a
    %   host array, or a gpuArray.
    
    %   Copyright 2016 The MathWorks, Inc

    properties
        % Value   The value of the learnable parameter
        %   The value of the learnable parameter during training. This can 
        %   be either a host array, or a gpuArray depending on what
        %   hardware resource we are using for training.
        Value

        % LearnRateFactor   Multiplier for the learning rate for this parameter
        %   A scalar double.
        LearnRateFactor

        % L2Factor   Multiplier for the L2 regularizer for this parameter
        %   A scalar double.
        L2Factor
    end
end