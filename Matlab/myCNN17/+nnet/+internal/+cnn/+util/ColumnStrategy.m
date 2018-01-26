classdef (Abstract) ColumnStrategy
    % ColumnStrategy   Column strategy interface
    
    %   Copyright 2016 The MathWorks, Inc.    
    
    properties (Abstract)
        % HorizontalBorder (char array)   Horizontal border of the table to
        %                                 print
        HorizontalBorder
        
        % Headings (char array)   Table headings
        Headings
        
        % Names (cellstr)   Names of MiniBatchSummary properties to be
        %                   reported
        Names
        
        % Formats (cellstr)   Formats to be used when printing the properties
        Formats
    end
end