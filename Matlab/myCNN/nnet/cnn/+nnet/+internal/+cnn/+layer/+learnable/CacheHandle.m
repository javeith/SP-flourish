classdef CacheHandle < handle
    % CacheHandle   Handle object used to cache values

    %   Copyright 2016 The MathWorks, Inc.
    
    properties
        Value = []
    end
    
    methods
        function this = CacheHandle(value)
            if nargin
                this.Value = value;
            else
                this.Value = [];
            end
        end
        
        function tf = isEmpty(this)
            tf = isempty(this.Value);
        end
    end
end