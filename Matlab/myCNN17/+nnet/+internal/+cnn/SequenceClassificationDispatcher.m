classdef SequenceClassificationDispatcher < nnet.internal.cnn.DataDispatcher
    % SequenceClassificationDispatcher   Dispatch time series data for
    % classification problems one mini batch at a time from a set of time
    % series
    %
    % Input data    - cell array of time series objects, where the length
    %               of the cell array is the number of observations. 
    %               Input predictors are numObs-by-1 cell arrays which
    %               contain sequences of size: DataSize-by-S
    %               Input responses are either:
    %                   - numObs-by-1 categorical arrays
    %                   - numObs-by-1 cell arrays which contain categorical
    %                   arrays of length 1-by-S. S here must match the S of
    %                   the corresponding predictor.
    %               
    % Output data   - numeric arrays with the following dimensions:
    %               Output predictors:
    %                   - DataSize-by-MiniBatchSize-by-S
    %               Output responses are either:
    %                   - ResponseSize-by-MiniBatchSize
    %                   - ResponseSize-by-MiniBatchSize-by-S
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % DataSize (int)   Number of dimensions per time step of the input
        %                   data (D)
        DataSize
        
        % ResponseSize (int)   Number of classes in the response
        ResponseSize
        
        % ResponseFormat (char)   Format of the response. Either:
        %       'full'  : classify the entire sequence. Response is
        %       numObs-by-1 categorical array.
        %       'step'  : classify each time step. Response size
        %       numObs-by-1 cell array, containing 1-by-SequenceLength
        %       categorical arrays.
        %       ''      : response is empty
        ResponseFormat
        
        % IsDone (logical)   True if there is no more data to dispatch
        IsDone
        
        % NumObservations (int)   Number of observations in the data set
        NumObservations
        
        % ClassNames (cellstr)   Array of class names corresponding to
        %            training data labels.
        ClassNames = {};
        
        % ResponseNames (cellstr)   Array of response names corresponding
        %               to training data response names. Since we cannot
        %               get the names anywhere for an array, we will use a
        %               fixed response name.
        ResponseNames = {'Response'};
        
        % SequenceLength (int)   
        SequenceLength
        
        % PaddingValue (scalar)
        PaddingValue
        
        % IsNextMiniBatchSameObs (logical)   False if the next mini-batch
        %                           corresponds to a new set of
        %                           observations. True if, for a given set
        %                           of observations, more data is to be
        %                           dispatched in the sequence dimension
        %                           before moving on to the next set of
        %                           observations.
        IsNextMiniBatchSameObs
        
        % ImageSize (Not used by this dispatcher)
        ImageSize
    end
    
    properties(Dependent)
        % MiniBatchSize (int)   Number of elements in a mini batch
        MiniBatchSize
    end
    
    properties
        % Precision   Precision used for dispatched data
        Precision
        
        % EndOfEpoch    Strategy to choose how to cope with a number of
        % observation that is not divisible by the desired number of mini
        % batches
        %
        % Allowed values: 'truncateLast', 'discardLast'
        EndOfEpoch
    end
    
    properties (Access = ?nnet.internal.cnn.DistributableDispatcher)
        
        % Data  (cell array)     A copy of the data in the workspace. This
        % is a numObservations-by-1 cell array, which for each observation
        % contains a DataSize-by-sequenceLength numeric array.
        Data
        
        % Response   A copy of the response data in the workspace. Either:
        % - numObservations-by-1 categorical vector 
        % - numObservations-by-1 cell array, which for each observation
        % contains a 1-by-sequenceLength categorical vector.
        Response
        
    end
    
    properties (Access = private)
        % StartIndexOfCurrentMiniBatch (int) Start index of current mini
        % batch
        StartIndexOfCurrentMiniBatch
        
        % EndIndexOfCurrentMiniBatch (int) End index of current mini batch
        EndIndexOfCurrentMiniBatch
        
        % OrderedIndices   Order to follow when indexing into the data.
        % This can keep a shuffled version of the indices.
        OrderedIndices
        
        % StartIndexOfCurrentSequence (int) Start index of the current
        % sequence
        StartIndexOfCurrentSequence
        
        % EndIndexOfCurrentSequence (int) End index of the current sequence
        EndIndexOfCurrentSequence
        
        % PrivateMiniBatchSize (int)   Number of elements in a mini batch
        PrivateMiniBatchSize
    end
    
    methods
        function this = SequenceClassificationDispatcher(data, response, miniBatchSize, ...
                sequenceLength, endOfEpoch, paddingValue, precision)
            % sequenceClassificationDispatcher   Constructor for sequence
            % classification generator
            %
            % data              - cell array of sequences for training. The
            %                   number of elements in the cell array is the
            %                   number of training observations. Each
            %                   sequence has dimension D x S. D is fixed
            %                   for all observations, but S may vary per
            %                   observation
            % response          - Data responses in the form of either:
            %                   numObservations x 1 categorical vector
            %                   numObservations x 1 cell array. 
            %                   The cell array must contain a 1 x S
            %                   categorical vector for each observation,
            %                   with S corresponding to the S of the data
            %                   at that observation
            % miniBatchSize     - Size of a mini batch expressed in number
            %                   of examples
            % sequenceLength    - Strategy to determine the length of the
            %                   sequences used per mini-batch. Options
            %                   are:
            %                   'shortest' to truncate all sequences in a
            %                   batch to the length of the shortest
            %                   sequence (default)
            %                   'longest' to pad all sequences in a
            %                   batch to the length of the longest sequence
            %                   Integer to pad or truncate all the
            %                   sequences in a batch to a specific integer
            %                   length.
            % endOfEpoch        - Strategy to choose how to cope with a
            %                   number of observations that is not
            %                   divisible by the desired number of mini
            %                   batches. One of: 
            %                   'truncateLast' to truncate the last mini
            %                   batch
            %                   'discardLast' to discard the last mini
            %                   batch (default)
            % paddingValue      - Scalar value used to pad sequences where
            %                   necessary. The default is 0.
            % precision         - What precision to use for the dispatched
            %                   data. Values are:
            %                   'single'
            %                   'double' (default).
            
            % Assign data and response
            [dataSize, numObservations, data] = iGetDataSize(data);
            this.Data = data;
            this.DataSize = dataSize;
            this.NumObservations = numObservations;
            [response, responseSize, classnames, responseFormat] = iResponseSizeAndNames(response);
            this.Response = response;
            this.ResponseSize = responseSize;
            this.ClassNames = classnames;
            this.ResponseFormat = responseFormat;
            
            % Assign properties
            this.SequenceLength = sequenceLength;
            this.EndOfEpoch = endOfEpoch;
            this.PaddingValue = paddingValue;
            this.Precision = precision;
            this.MiniBatchSize = miniBatchSize;
            this.OrderedIndices = 1:this.NumObservations;
        end
        
        function [miniBatchData, miniBatchResponse, miniBatchIndices] = next(this)
            % next   Get the data and response for the next mini batch and
            % correspondent indices
            
            % Map the indices into data and sequence dimension
            [miniBatchIndices, sequenceIndices] = this.computeIndices();
            
            % Read the data
            [miniBatchData, miniBatchResponse, advanceSequence] = this.readData(miniBatchIndices, sequenceIndices);
            
            if any(advanceSequence)
                this.advanceCurrentSequenceIndices();
                this.IsNextMiniBatchSameObs = true;
            else
                this.resetCurrentSequenceIndices();
                this.IsNextMiniBatchSameObs = false;
                % Advance indices of current mini batch
                this.advanceCurrentMiniBatchIndices();
            end
        end
        
        function start(this)
            % start   Go to first mini batch
            this.IsDone = false;
            this.IsNextMiniBatchSameObs = false;
            this.StartIndexOfCurrentMiniBatch = 1;
            this.EndIndexOfCurrentMiniBatch = this.MiniBatchSize;
            
            this.resetCurrentSequenceIndices();
        end
        
        function shuffle(this)
            % shuffle   Shuffle the data
            this.OrderedIndices = randperm(this.NumObservations);
        end
        
        function value = get.MiniBatchSize(this)
            value = this.PrivateMiniBatchSize;
        end
        
        function set.MiniBatchSize(this, value)
            value = min(value, this.NumObservations);
            this.PrivateMiniBatchSize = value;
        end
    end
    
    methods (Access = private)
        function advanceCurrentMiniBatchIndices(this)
            % advanceCurrentMiniBatchIndices   Move forward start and end
            % index of current mini batch based on mini batch size
            if this.EndIndexOfCurrentMiniBatch == this.NumObservations
                % We are at the end of a cycle
                this.IsDone = true;
            elseif this.EndIndexOfCurrentMiniBatch + this.MiniBatchSize > this.NumObservations
                % Last mini batch is smaller
                if isequal(this.EndOfEpoch, 'truncateLast')
                    this.StartIndexOfCurrentMiniBatch = this.StartIndexOfCurrentMiniBatch + this.MiniBatchSize;
                    this.EndIndexOfCurrentMiniBatch = this.NumObservations;
                else % discardLast
                    % Move the starting index after the end, so that the
                    % dispatcher will return empty data
                    this.StartIndexOfCurrentMiniBatch = this.EndIndexOfCurrentMiniBatch+1;
                    this.IsDone = true;
                end
            else
                % We are in the middle of a cycle
                this.StartIndexOfCurrentMiniBatch = this.StartIndexOfCurrentMiniBatch + this.MiniBatchSize;
                this.EndIndexOfCurrentMiniBatch = this.EndIndexOfCurrentMiniBatch + this.MiniBatchSize;
            end
        end
        
        function advanceCurrentSequenceIndices(this)
            % advanceCurrentSequenceIndices   Move forward start and end
            % index in the sequence dimension
            this.StartIndexOfCurrentSequence = this.StartIndexOfCurrentSequence + this.SequenceLength;
            this.EndIndexOfCurrentSequence = this.EndIndexOfCurrentSequence + this.SequenceLength;
        end
        
        function resetCurrentSequenceIndices(this)
            % resetCurrentSequenceIndices   Take sequence indices back to
            % their initial values
            this.StartIndexOfCurrentSequence = 1;
            if isnumeric(this.SequenceLength)
                this.EndIndexOfCurrentSequence = this.SequenceLength;
            else
                this.EndIndexOfCurrentSequence = [];
            end
        end
        
        function [miniBatchData, miniBatchResponse, advanceSeq] = readData(this, indices, seqIndices)
            % readData  Read data and response corresponding to indices
            if strcmp( this.SequenceLength, 'shortest' )
                [miniBatchData, miniBatchResponse, advanceSeq] = this.readShortestBatchAndResponse(indices);
            elseif strcmp( this.SequenceLength, 'longest' )
                [miniBatchData, miniBatchResponse, advanceSeq] = this.readLongestBatchAndResponse(indices);
            elseif isnumeric( this.SequenceLength )
                [miniBatchData, miniBatchResponse, advanceSeq] = this.readFixedBatchAndResponse(indices, seqIndices);
            else
                % Erroneous SequenceLength value
            end
            % Cast the data
            miniBatchData = this.Precision.cast( miniBatchData );
            % Cast the response
            miniBatchResponse = this.Precision.cast( miniBatchResponse );
        end
        
        function [miniBatchData, miniBatchResponse, advanceSeq] = readShortestBatchAndResponse(this, indices)
            % readShortestBatchAndResponse   Create mini-batches and
            % response where the sequence length is truncated to the length
            % of the shortest sequence in the batch.
            
            % advanceSeq is always false. Since we are truncating at
            % the shortest sequence length, there is always one
            % observation in the mini-batch which is a complete
            % sequence. Advancing the sequence would lead to a zero
            % observation, which is not allowed
            advanceSeq = false;
            batchSize = numel(indices);
            % Get data sequence dimensions
            dataSeqLengths = cellfun( @(x)size(x, 2), this.Data( indices ) );
            shortestSeq = min( dataSeqLengths );
            % Initialize mini-batches
            miniBatchData = this.initializeMiniBatchData( batchSize, shortestSeq );
            % Allocate mini-batch data and mini-batch response
            if strcmp( this.ResponseFormat, 'step' )
                % If response is step-wise, gather mini-batch response for
                % each observation
                miniBatchResponse = this.initializeMiniBatchResponse( batchSize, shortestSeq );
                for ii = 1:batchSize
                    % Get the data
                    dataSeq = this.Data{ indices(ii) };
                    % Allocate data into mini-batch
                    miniBatchData(:, ii, :) = dataSeq(:, 1:shortestSeq);
                    % Allocate response
                    responseSeq = this.Response{ indices(ii) };
                    miniBatchResponse(:, ii, :) = this.dummify( responseSeq(:, 1:shortestSeq) );
                end
            else
                % If response is not step-wise, create mini-batch response
                % with readResponses() method
                for ii = 1:batchSize
                    % Get the data
                    dataSeq = this.Data{ indices(ii) };
                    % Allocate data into mini-batch
                    miniBatchData(:, ii, :) = dataSeq(:, 1:shortestSeq);
                end
                % Read responses
                miniBatchResponse = this.readResponses( indices );
            end
        end
        
        function [miniBatchData, miniBatchResponse, advanceSeq] = readLongestBatchAndResponse(this, indices)
            % readLongestBatchAndResponse   Create mini-batches and
            % response where the sequence length is padded to the length of
            % the longest sequence in the batch
            
            % advanceSeq is always false. Since we are truncating at
            % the longest sequence length, there is never a need to
            % advance in the sequence dimension
            advanceSeq = false;
            batchSize = numel(indices);
            % Get data sequence dimensions
            dataSeqLengths = cellfun( @(x)size(x, 2), this.Data( indices ) );
            longestSeq = max( dataSeqLengths );
            % Initialize mini-batch
            miniBatchData = this.initializeMiniBatchData( batchSize, longestSeq );
            % Allocate mini-batch data and mini-batch response
            if strcmp( this.ResponseFormat, 'step' )
                % If response is step-wise, gather mini-batch response for
                % each observation
                miniBatchResponse = this.initializeMiniBatchResponse( batchSize, longestSeq );
                for ii = 1:batchSize
                    % Get the data
                    dataSeq = this.Data{ indices(ii) };
                    % Allocate data into mini-batch
                    miniBatchData(:, ii, 1:dataSeqLengths(ii)) = dataSeq;
                    % Allocate response
                    responseSeq = this.Response{ indices(ii) };
                    miniBatchResponse(:, ii, 1:dataSeqLengths(ii)) = this.dummify( responseSeq );
                end
            else
                % If response is not step-wise, create mini-batch response
                % with readResponses() method
                for ii = 1:batchSize
                    % Get the data
                    dataSeq = this.Data{ indices(ii) };
                    % Allocate data into mini-batch
                    miniBatchData(:, ii, 1:dataSeqLengths(ii)) = dataSeq;
                end
                % Read responses
                miniBatchResponse = this.readResponses( indices );
            end
        end
        
        function [miniBatchData, miniBatchResponse, advanceSeq] = readFixedBatchAndResponse(this, indices, seqIndices)
            % readFixedBatchAndResponse   Create a mini-batch and response
            % where the sequence length is padded/truncated to a specified
            % length
            
            % Initialize mini-batch data
            batchSize = numel(indices);
            miniBatchData = this.initializeMiniBatchData( batchSize, this.SequenceLength );
            % Get sequence dimensions
            dataSeqLengths = cellfun( @(x)size(x, 2), this.Data( indices ) );
            % Determine whether another pass along sequence dimension is
            % needed
            advanceSeq = max( dataSeqLengths ) > max( seqIndices );
            % Allocate mini-batch data and mini-batch response
            if strcmp( this.ResponseFormat, 'step' )
                % If response is step-wise, gather mini-batch response for
                % each observation
                miniBatchResponse = this.initializeMiniBatchResponse( batchSize, this.SequenceLength );
                for ii = 1:batchSize
                    % Get the data
                    [dataSeq, dataInds, batchInds] = this.getMiniBatchDataPerObs(indices(ii), seqIndices, dataSeqLengths(ii));
                    % Write into the mini-batch
                    miniBatchData(:, ii, batchInds) = dataSeq(:, dataInds);
                    % Allocate response
                    responseSeq = this.Response{ indices(ii) };
                    miniBatchResponse(:, ii, batchInds) = this.dummify( responseSeq(:, dataInds) );
                end
            else
                % If response is not step-wise, create mini-batch response
                % with readResponses() method
                for ii = 1:batchSize
                    % Get the data
                    [dataSeq, dataInds, batchInds] = this.getMiniBatchDataPerObs(indices(ii), seqIndices, dataSeqLengths(ii));
                    % Write into the mini-batch
                    miniBatchData(:, ii, batchInds) = dataSeq(:, dataInds);
                end
                % Read responses
                miniBatchResponse = this.readResponses( indices );
            end
        end
        
        function [data, dataInds, batchInds] = getMiniBatchDataPerObs(this, index, seqIndices, dataSeqLength)
            % Get the data
            data = this.Data{ index };
            % Get indices for indexing into the data
            dataInds = seqIndices(1):min(seqIndices(end), dataSeqLength);
            % Get indices for indexing into the mini-batch
            batchInds = 1:numel(dataInds);
        end
        
        function responses = readResponses(this, indices)
            if isempty(this.Response)
                responses = [];
            elseif iscategorical( this.Response )
                % Categorical vector of responses
                responses = this.dummify( this.Response(indices) );
            elseif iscell( this.Response )
                % Cell array of categorical responses
            end
        end
        
        function [dataIndices, seqIndices] = computeIndices(this)
            % computeIndices    Compute the indices into the data from
            % start and end index, and compute sequence indices
            
            dataIndices = this.StartIndexOfCurrentMiniBatch:this.EndIndexOfCurrentMiniBatch;
            
            % Convert sequential indices to ordered (possibly shuffled) indices
            dataIndices = this.OrderedIndices(dataIndices);
            
            % Compute sequence indices
            seqIndices = this.StartIndexOfCurrentSequence:this.EndIndexOfCurrentSequence;
        end
        
        function miniBatchData = initializeMiniBatchData(this, batchSize, sequenceLength )
            % initializeMiniBatchData   Create an array which will have the
            % mini-batch data assigned into it
            miniBatchData = this.PaddingValue.*ones( this.DataSize, batchSize, sequenceLength );
        end
        
        function miniBatchResponse = initializeMiniBatchResponse(this, batchSize, sequenceLength )
            % initializeMiniBatchData   Create an array which will have the
            % mini-batch data assigned into it
            miniBatchResponse = this.PaddingValue.*ones( this.ResponseSize, batchSize, sequenceLength );
        end
        
        function paddedData = padData(this, data, seqIndices, seqComplement )
            % padData   Add a specified amount of padding to the data
            paddedData = cat(2, data(:, seqIndices), this.PaddingValue.*ones(this.DataSize, numel(seqComplement)));
        end
        
        function dummy = dummify(this, categoricalIn)
            % dummify   Dummify a categorical vector of size numObservations x 1 to
            % return a matrix of size numClasses x numObservations
            if isempty( categoricalIn )
                numClasses = 0;
                numObs = 1;
                dummy = zeros( numClasses, numObs );
            else
                categoricalIn = reordercats( categoricalIn, this.ClassNames );
                numObservations = numel(categoricalIn);
                numCategories = this.ResponseSize;
                dummifiedSize = [numCategories, numObservations];
                dummy = zeros(dummifiedSize);
                categoricalIn = reshape( categoricalIn, 1, numel( categoricalIn ) );
                idx = sub2ind(dummifiedSize, single(categoricalIn), 1:numObservations);
                idx(isnan(idx)) = [];
                dummy(idx) = 1;
            end
        end
    end
end

function [dataDimension, numObservations, data] = iGetDataSize(data)
if iscell( data )
    numObservations = numel( data );
    dataDimension = size( data{1}, 1 );
elseif isnumeric( data )
    dims = size( data );
    dataDimension = dims(1);
    numObservations = 1;
    % Wrap data into cell
    data = { data };
end
end

function [response, responseSize, classnames, responseFormat] = iResponseSizeAndNames(response)
% Convert single observation seq2seq response to cell
if iscategorical( response ) && isrow( response )
    response = { response };
end

if isempty( response )
    % Empty response => prediction only
    classnames = {};
    responseSize = [];
    responseFormat = '';
    
elseif iscategorical( response )
    % Categorical response => seq2one problem
    allResponses = response;
    responseFormat = 'full';
    classnames = categories( allResponses );
    responseSize = numel( classnames );
    
elseif iscell( response )
    % Cell array of categoricals => seq2seq problem
    allResponses = [ response{:} ];
    responseFormat = 'step';
    classnames = categories( allResponses );
    responseSize = numel( classnames );
    for ii = 1:numel( response )
        response{ii} = setcats( response{ii}, classnames );
    end
end
end