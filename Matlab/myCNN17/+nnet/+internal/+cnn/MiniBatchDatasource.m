classdef MiniBatchDatasource < handle
    % MiniBatchDatasource  Abstract superclass to define interface for
    % extracting 4D data mini-batches from an input data set.
    
    % Copyright 2017 The MathWorks, Inc.

    properties (Abstract)
        % MiniBatchSize (int)  Number of elements in a mini batch
        MiniBatchSize
        
        % NumberOfObservations (int)  Number of total observations in one
        % epoch.
        NumberOfObservations
    end
    
    methods (Abstract)
        [X,Y] = nextBatch(self); % Defines next batch and increments internal iterator state
        reset(self); % Reset iterator state to first batch
    end
    
end
