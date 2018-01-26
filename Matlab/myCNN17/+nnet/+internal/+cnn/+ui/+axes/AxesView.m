classdef(Abstract) AxesView < handle
    % AxesView   Interface for view of axes
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Abstract, SetAccess = private)
        % Panel   (uipanel) The parent panel of the axes
        Panel
    end
    
    methods(Abstract)
        update(this)
        % update the view 
    end
end

