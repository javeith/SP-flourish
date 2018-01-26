classdef DistributableFilePathTableMiniBatchDatasource < nnet.internal.cnn.DistributableMiniBatchDatasource
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Constant)
        CanPreserveOrder = true;
    end
    
    methods
        
        function [distributedData, subBatchSizes] = distribute( this, proportions )
        % distribute   Split the dispatcher into partitions according to
        % the given proportions
            
            % Create a cell containing one partition of the data per
            % entry in the proportions array
            [ distributedTableData, subBatchSizes ] = ...
                iSplitTable( this.TableData, this.MiniBatchSize, proportions );
            
            % Create a new FilePathTableDispatcher wrapping each of those
            % tables.
            % Note we always use 'truncateLast' for the endOfEpoch
            % parameters and instead deal with this in the Trainer.
            numPartitions = numel(proportions);
            distributedData = cell(numPartitions, 1);
            for p = 1:numPartitions
                distributedData{p} = nnet.internal.cnn.FilePathTableMiniBatchDatasource( ...
                    distributedTableData{p}, ...
                    subBatchSizes(p));
            end
        
        end

    end
end

function [tablePartitions, subBatchSizes] = iSplitTable( ...
    tableData, miniBatchSize, proportions )
% Divide up table by rows according to the given proportions

import nnet.internal.cnn.DistributableMiniBatchDatasource.interleavedSelectionByWeight;

numObservations= height(tableData);
numWorkers = numel(proportions);
tablePartitions = cell(numWorkers, 1);

% Get the list of indices into the data for each partition
[indicesCellArray, subBatchSizes] = interleavedSelectionByWeight( ...
    numObservations, miniBatchSize, proportions );

% Loop through indexing into the data to create the partitions
for p = 1:numWorkers
    if subBatchSizes(p) > 0
        % Index data to create partition
        tablePartitions{p} = tableData(indicesCellArray{p}, :);
    end
end
end