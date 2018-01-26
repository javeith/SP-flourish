classdef TrainingDynamicParameter < nnet.internal.cnn.layer.dynamic.DynamicParameter
    % TrainingDynamicParameter   Dynamic parameter for use at training time
    %
    %   This class is used to represent dynamic parameters during
    %   training. The representation that is used is very simple. The
    %   parameter is stored in the property Value, which can be a
    %   host array, or a gpuArray.
    
    %   Copyright 2017 The MathWorks, Inc

    properties
        % Value   The value of the dynamic parameter
        %   The value of the dynamic parameter during training. This can 
        %   be either a host array, or a gpuArray depending on what
        %   hardware resource we are using for training.
        Value

        % Remember   Logical which is true when the parameter is to be
        % remebered
        Remember
    end
end