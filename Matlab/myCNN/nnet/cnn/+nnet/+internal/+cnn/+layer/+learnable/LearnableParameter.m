classdef LearnableParameter
    % LearnableParameter   Interface for learnable parameters

    %   Copyright 2015-2016 The MathWorks, Inc

    properties(Abstract)
        % Value   The value of the learnable parameter
        %   An array which can be a gpuArray or a host array.
        Value

        % LearnRateFactor   Multiplier for the learning rate for this parameter
        %   A scalar double.
        LearnRateFactor

        % L2Factor   Multiplier for the L2 regularizer for this parameter
        %   A scalar double.
        L2Factor
    end
end