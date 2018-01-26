classdef DistributedDispatcher < nnet.internal.cnn.DataDispatcher
    % DistributedDispatcher Dispatcher split onto workers of a parallel
    % pool.
    %
    % Objects of this class reside on the CLIENT, only the Composite
    % property DistributedData can actually be passed into the pool.
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % ImageSize  (1x3 int) Size of each image to be dispatched
        ImageSize
        
        % ResponseSize   (1x3 int) Size of each response to be dispatched
        ResponseSize
        
        % NumObservations   (int) Number of observations in the data set
        NumObservations
        
        % IsDone (logical)  Unneeded Abstract property that must be
        % overloaded
        IsDone
        
        % ClassNames (cellstr) Array of class names corresponding to
        %            training data labels.
        ClassNames
        
        % ResponseNames (cellstr) Array of response names corresponding to
        %               training data response names.
        ResponseNames    
    end
    
    properties               
        % MiniBatchSize (int)   Number of elements in a mini batch
        MiniBatchSize

        % Precision   Precision used for dispatched data
        Precision
        
        % EndOfEpoch    Strategy to choose how to cope with a number of
        % observations that is not divisible by the desired number of mini
        % batches
        %
        % Allowed values: 'truncateLast', 'discardLast'
        EndOfEpoch
    end
    
    properties (SetAccess = private)
        % Proportions (double)  Proportion of data on each worker
        Proportions
        
        % SubBatchSizes (double)  Size of the portion of a minibatch
        % processed on each worker (sums to MiniBatchSize)
        SubBatchSizes
    end
    
    properties (SetAccess = private)
        % DistributedData (Composite) Dispatcher on each worker, each with
        % a partition of the data
        DistributedData
    end
    
    methods
        function this = DistributedDispatcher(data, workerLoad, preserveDataOrder)
            % DistributedDispatcher   Constructor for a distributed
            % dispatcher based on an input data dispatcher
            %
            % data            - The DataDispatcher to be distributed
            % workerLoad      - Proportions of the data to be processed
            %                 by each worker, array of length numlabs.
            %                 Elements can be zero, meaning this worker
            %                 does nothing
            % preserveDataOrder - Ensure that the order that data will be
            %                 read is unchanged by distribution
            if ~isa( data, 'nnet.internal.cnn.DistributableDispatcher' )
                error(message('nnet_cnn:internal:cnn:DistributedDispatcher:DispatcherNotDistributable', class( data )));
            end
            if preserveDataOrder && ~data.CanPreserveOrder
                error(message('nnet_cnn:internal:cnn:DistributedDispatcher:DispatcherCannotPreserveOrder', class( data )));
            end
            
            this.NumObservations = data.NumObservations;
            this.ClassNames = data.ClassNames;
            this.ResponseNames = data.ResponseNames;
            this.MiniBatchSize = data.MiniBatchSize;
            this.EndOfEpoch = data.EndOfEpoch;
            this.Precision = data.Precision;
            this.ImageSize = data.ImageSize;
            this.ResponseSize = data.ResponseSize;
            
            this.Proportions = iCalculateProportions( workerLoad );
            
            % The best way to ensure zero bias towards any one grouping of
            % similar observations on any one worker is to shuffle the data
            % before distribution.
            if ~preserveDataOrder
                data.shuffle();
            end
            
            % Distribute data to workers in the specified proportions
            [distributedDataCell, this.SubBatchSizes] = distribute( data, this.Proportions );
            this.DistributedData = Composite();
            [ this.DistributedData{:} ] = deal( distributedDataCell{:} );
        end
        
        % next  Declare but do not implement, not meaningful to call on the
        % client
        next(~)
        
        % start  Declare but do not implement, not meaningful to call on the
        % client
        start(~)
        
        % shuffle  Declare but do not implement, not meaningful to call on
        % the client
        shuffle(~)

    end
end

function proportions = iCalculateProportions( workerLoad )
totalWork = sum( workerLoad );
assert( totalWork > 0 );
proportions = workerLoad / totalWork;
end
