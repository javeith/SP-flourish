classdef DistributableMiniBatchDatasourceDispatcher < nnet.internal.cnn.DistributableDispatcher
    % DistributableMiniBatchDatasourceDispatcher DistributableDispatcher
    % implementation for MiniBatchDatasource Data Dispatchers.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Constant)
        CanPreserveOrder = true;
    end
    
    methods
        
        function [distributedData, subBatchSizes] = distribute( this, proportions )
            % distribute   Split the dispatcher into partitions according to
            % the given proportions
            
            % Create a cell array of MiniBatchDatasources containing one portion of
            % the input MiniBatchDatasource per entry in the partitions array.
            [ dsPartitions, subBatchSizes ] = distribute( this.Datasource, proportions );
            
            % Create a MiniBatchDatasourceDispatcher containing each of those
            % MiniBatchDatasources.
            % Note we always use 'truncateLast' for the endOfEpoch
            % parameters and instead deal with this in the Trainer.
            numPartitions = numel(proportions);
            distributedData = cell(numPartitions, 1);
            for p = 1:numPartitions
                distributedData{p} = nnet.internal.cnn.MiniBatchDatasourceDispatcher( ...
                    dsPartitions{p}, ...
                    subBatchSizes(p), 'truncateLast', this.Precision);
            end
        end
        
    end
    
end