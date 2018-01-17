classdef Precision
    % Preicsion     Class to handle data precision
    
    %   Copyright 2015 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % Type   Type of precision to be used.
        %        One of:
        %        'single'
        %        'double'
        Type
    end
    
    methods
        function this = Precision(type)
            this.Type = type;
        end
        
        function data = cast(this, data)
            % cast   Cast floating point data using the precision specified
            % at construction time
            data = cast(data, this.Type);
        end
    end
end
