classdef BackgroundDispatchableDatasource < handle
   
    properties (Hidden)
        UseParallel = false;
        RunInBackgroundOnAuto = false;
    end
    
    methods (Hidden)
        
        function [data, response] = getObservations(this, indices) %#ok<STOUT,INUSD>
            % getObservations  Get a batch of observations as specified by
            % their indices. The base class version asserts and should never be
            % called.
            assert( false, 'MiniBatchDatasource has no implementation of getObservations method' );
        end
        
        function [miniBatchData, miniBatchResponse] = getBatch(this, batchIndex)
            % getBatch   Get the data and response for a specific mini batch
            % and corresponding indices. Base class implementation uses
            % getObservations but can be overloaded.
            
            % Work out which observations go with this batch
            miniBatchStartIndex = ( (batchIndex - 1) * this.MiniBatchSize ) + 1;
            miniBatchEndIndex = min( this.NumberOfObservations, miniBatchStartIndex + this.MiniBatchSize - 1 );
            miniBatchIndices = miniBatchStartIndex:miniBatchEndIndex;
            
            % Read the data
            [miniBatchData, miniBatchResponse] = this.getObservations(miniBatchIndices);
        end
        
    end
    
end