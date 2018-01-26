classdef (Abstract) ClassificationLayer < nnet.internal.cnn.layer.OutputLayer
    % OutputLayer     Interface for convolutional neural network
    % classification output layers
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Abstract)
        % ClassNames   The names of the classes
        ClassNames
    end
    
    properties (Abstract, SetAccess = private)
        % NumClasses   Number of classes
        NumClasses
    end
end
