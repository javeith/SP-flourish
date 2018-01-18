classdef FourDArrayDispatcher < nnet.internal.cnn.DataDispatcher
    % FourDArrayDispatcher class to dispatch 4D data one mini batch at a
    %   time from 4D numeric data
    %
    % Input data    - 4D data where the last dimension is the number of
    %               observations.
    % Output data   - 4D data where the last dimension is the number of
    %               observations in that mini batch. The type of the data
    %               in output will be the same as the one in input
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % ImageSize  (1x3 int) Size of each image to be dispatched
        ImageSize
        
        % IsDone (logical)     True if there is no more data to dispatch
        IsDone
        
        % NumObservations (int) Number of observations in the data set
        NumObservations
        
        % ClassNames (cellstr) Array of class names corresponding to
        %            training data labels.
        ClassNames
        
        % PrivateMiniBatchSize (int)   Number of elements in a mini batch
        PrivateMiniBatchSize
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
    
    properties (Access = private)
        % StartIndexOfCurrentMiniBatch (int) Start index of current mini
        % batch
        StartIndexOfCurrentMiniBatch
        
        % EndIndexOfCurrentMiniBatch (int) End index of current mini batch
        EndIndexOfCurrentMiniBatch
        
        % Data  (array)     A copy of the data in the workspace. This is a
        % 4-D array where last dimension indicates the number of examples
        Data
        
        % Response (vector) A copy of the response data in the workspace.
        % Each column is a different response
        Response             
        
        % OrderedIndices   Order to follow when indexing into the data.
        % This can keep a shuffled version of the indices.
        OrderedIndices
    end       
    
    methods
        function this = FourDArrayDispatcher(data, response, miniBatchSize, endOfEpoch, precision)
            % FourDArrayDispatcher   Constructor for 4-D array data dispatcher
            %
            % data              - 4D array from the workspace where the last
            %                   dimension is the number of observations
            % response          - A column vector of categorical responses
            % miniBatchSize     - Size of a mini batch express in number of
            %                   examples
            % endOfEpoch        - Strategy to choose how to cope with a
            %                   number of observation that is not divisible
            %                   by the desired number of mini batches
            %                   One of:
            %                   'truncateLast' to truncate the last mini
            %                   batch
            %                   'discardLast' to discard the last mini
            %                   batch
            % precision         - What precision to use for the dispatched
            %                   data
            this.Data = data;
            this.Response = response;
            this.ClassNames = iGetClassNames(this.Response);
            this.ImageSize = iGetImageDimensionsFromArray(data);
            this.NumObservations = size(data, 4);
            this.MiniBatchSize = miniBatchSize;
            assert(isequal(endOfEpoch,'truncateLast') || isequal(endOfEpoch,'discardLast'), 'nnet.internal.cnn.FourDArrayDispatcher error: endOfEpoch should be one of ''truncateLast'', ''discardLast''.');
            this.EndOfEpoch = endOfEpoch;
            this.Precision = precision;
            this.OrderedIndices = 1:this.NumObservations;
        end
        
        function [miniBatchData, miniBatchResponse, miniBatchIndices] = next(this)
            % next   Get the data and response for the next mini batch and
            % correspondent indices
            
            % Map the indices into data
            miniBatchIndices = this.computeDataIndices();
            
            % Read the data
            [miniBatchData, miniBatchResponse] = this.readData(miniBatchIndices);
            
            % Advance indices of current mini batch
            this.advanceCurrentMiniBatchIndices();
        end
        
        function start(this)
            % start     Go to first mini batch
            this.IsDone = false;
            this.StartIndexOfCurrentMiniBatch = 1;                       
                        
            this.EndIndexOfCurrentMiniBatch = this.MiniBatchSize;
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
                    this.IsDone = true;
                end
            else
                % We are in the middle of a cycle
                this.StartIndexOfCurrentMiniBatch = this.StartIndexOfCurrentMiniBatch + this.MiniBatchSize;
                this.EndIndexOfCurrentMiniBatch = this.EndIndexOfCurrentMiniBatch + this.MiniBatchSize;
            end
        end
        
        function [miniBatchData, miniBatchResponse] = readData(this, indices)
            % readData  Read data and response corresponding to indices
            miniBatchData = this.Precision.cast( this.Data(:,:,:,indices) );
            if isempty(this.Response)
                miniBatchResponse = [];
            else
                miniBatchResponse = this.Precision.cast( iDummify( this.Response(indices) ) );
            end
        end
        
        function dataIndices = computeDataIndices(this)
            % computeDataIndices    Compute the indices into the data from
            % start and end index
            
            dataIndices = this.StartIndexOfCurrentMiniBatch:this.EndIndexOfCurrentMiniBatch;
            
            % Convert sequential indices to ordered (possibly shuffled) indices
            dataIndices = this.OrderedIndices(dataIndices);
        end
    end
end

function dummy = iDummify(categoricalIn)
if isempty( categoricalIn )
    numClasses = 0;
    numObs = 1;
    dummy = zeros( 1, 1, numClasses, numObs );
else
    dummy = nnet.internal.cnn.util.dummify(categoricalIn);
end
end

function classnames = iGetClassNames(response)
if isa(response, 'categorical')
    classnames = categories( response );
else
    classnames = {};
end
end

function imageDimensions = iGetImageDimensionsFromArray(data)
dataSize = size(data);
heightAndWidth = dataSize(1:2);
if(ismatrix(data))
    numChannels = 1;
else
    numChannels = dataSize(3);
end
imageDimensions = [heightAndWidth numChannels];
end