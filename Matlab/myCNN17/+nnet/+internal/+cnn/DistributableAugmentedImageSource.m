classdef DistributableAugmentedImageSource < nnet.internal.cnn.DistributableMiniBatchDatasource
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Hidden, Constant)
        CanPreserveOrder = true
    end
    
    methods (Hidden)
        
        function [distributedData, subBatchSizes] = distribute( this, proportions )
            % distribute   Split the dispatcher into partitions according to
            % the given proportions
            
            % Create a cell array of MiniBatchDatasources containing one portion of
            % the input MiniBatchDatasource per entry in the partitions array.
            [ dsPartitions, subBatchSizes ] = distribute( this.DatasourceInternal, proportions );
            
            numPartitions = numel(proportions);
            distributedData = cell(numPartitions, 1);
            for p = 1:numPartitions
                partitionedAugmentedImageDatasource = augmentedImageSource(this.OutputSize,dsPartitions{p},...
                    'BackgroundExecution',this.UseParallel,...
                    'DataAugmentation',this.DataAugmentation,...
                    'OutputSizeMode',this.OutputSizeMode,...
                    'ColorPreprocessing',this.ColorPreprocessing);
                    
                partitionedAugmentedImageDatasource.MiniBatchSize = subBatchSizes(p);
                distributedData{p} = partitionedAugmentedImageDatasource;
            end
        end
    end
end


