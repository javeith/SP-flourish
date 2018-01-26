classdef(Abstract) UpdateableMetric < handle
    % UpdateableMetric   Interface for metrics which can be updated from a struct of information.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods(Abstract)
        update(this, infoStruct)
        % update   Updates the metric from the information given in the
        % infoStruct.
    end
    
end

