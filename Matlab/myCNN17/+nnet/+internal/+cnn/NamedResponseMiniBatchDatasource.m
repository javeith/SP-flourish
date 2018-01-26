classdef NamedResponseMiniBatchDatasource < handle
    % NamedResponseMiniBatchDatasource Abstract interface for declaring that a MiniBatchDatasource has named responses.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Abstract)
       ResponseNames 
    end
    
end