classdef MiniBatchDatasourceDispatcher < nnet.internal.cnn.DataDispatcher &...
        nnet.internal.cnn.BackgroundCapableDispatcher &...
        nnet.internal.cnn.DistributableMiniBatchDatasourceDispatcher
    
    % MiniBatchDatasourceDispatcher class to dispatch 4D data one mini batch at a
    %   time from a MiniBatchDatasource
    %
    % Input data    - MiniBatchDatasource.
    %
    % Output data   - 4D data where the last dimension is the number of
    %               observations in that mini batch. The type of the data
    %               in output will be the same as the one in input
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % ImageSize  (1x3 int) Size of each image to be dispatched
        ImageSize
        
        % ResponseSize   (1x3 int) Size of each response to be dispatched
        ResponseSize
        
        % IsDone (logical)     True if there is no more data to dispatch
        IsDone
        
        % NumObservations (int) Number of observations in the data set
        NumObservations
        
        % ClassNames (cellstr) Array of class names corresponding to
        %            training data labels.
        ClassNames = {};
        
        % ResponseNames (cellstr) Array of response names corresponding to
        %               training data response names. Since we cannot get
        %               the names anywhere for an array, we will use a
        %               fixed response name.
        ResponseNames = {'Response'};
        
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
        % Datasource  The MiniBatchDatasource we are going to
        % read data and responses from
        Datasource
    end
    
    properties (Access = private)
        
        % CurrentStartIndex  (int)   Current index of image to be dispatched
        CurrentStartIndex
        
        InitialMiniBatchSize % Initial mini-batch size
        
    end
    
    methods
        function this = MiniBatchDatasourceDispatcher(datasource, miniBatchSize, endOfEpoch, precision)
            % FourDArrayDispatcher   Constructor for 4-D array data dispatcher
            %
            % data              - 4D array from the workspace where the last
            %                   dimension is the number of observations
            % response          - Data responses in the form of:
            %                   numObservations x 1 categorical vector
            %                   numObservations x numResponses numeric matrix
            %                   H x W x C x numObservations numeric tensor
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
            
            this.EndOfEpoch = endOfEpoch;
            this.Precision = precision;
            this.IsDone = false;
            
            this.Datasource = datasource;

            if isempty( datasource )
                this.NumObservations = 0;                
            else
                                
                % Count the number of observations
                this.NumObservations = this.Datasource.NumberOfObservations;
                
                [this.MiniBatchSize,this.InitialMiniBatchSize] = deal(miniBatchSize);
                
                % Get example
                [exampleImage,exampleResponse,this.ClassNames] = getExampleInfoFromDatasource(this);
                
                % Set the expected image size
                this.ImageSize = iImageSize( exampleImage );
                
                % Set the expected response size to be dispatched
                this.ResponseSize = iResponseSize(exampleResponse);
                
                % NamedResponseMiniBatchDatasource is internal interface used to
                % subscribe to non-default response naming in output layer.
                if isa(this.Datasource,'nnet.internal.cnn.NamedResponseMiniBatchDatasource')
                    this.ResponseNames = this.Datasource.ResponseNames;
                end
                                
                % Run in background if a Datasource subscribes to
                % BackgroundDispatchableDatasource and UseParallel is
                % set to true.
                if isa(this.Datasource,'nnet.internal.cnn.BackgroundDispatchableDatasource')
                    this.RunInBackgroundOnAuto = this.Datasource.RunInBackgroundOnAuto;
                    this.setRunInBackground(this.Datasource.UseParallel);
                end
                
            end
        end
        
        
        function [miniBatchData, miniBatchResponse, miniBatchIndices] = next(this)
            % next   Get the data and response for the next mini batch and
            % correspondent indices
            
            % Return next batch of data
            if ~this.IsDone
                [miniBatchData, miniBatchResponse] = readData(this);
                miniBatchIndices = this.currentIndices();
            else
                miniBatchIndices = [];
                [miniBatchData,miniBatchResponse] = deal(this.Precision.cast([]));
            end
            
            if isempty(miniBatchIndices)
                return
            end
            
            this.advanceCurrentStartIndex(this.MiniBatchSize);
            
            nextMiniBatchSize = this.nextMiniBatchSize(miniBatchIndices(end)); % Manage discard vs. truncate batch behavior.
            if nextMiniBatchSize > 0
                this.MiniBatchSize = nextMiniBatchSize;
            else
                this.IsDone = true;
            end
            
        end
        
        function [miniBatchData, miniBatchResponse, indicesOut] = getObservations(this,indices)
            % getObservations  Overload of method to retrieve specific
            % observations
            
            indicesOut = indices;
            if isempty(indices)
                miniBatchData = [];
                miniBatchResponse = [];
            else
                [X, Y] =  this.Datasource.getObservations(indices);
                miniBatchData = this.readObservations(X);
                miniBatchResponse = this.readResponses(Y);
            end
            
        end
                
        function start(this)
            % start     Go to first mini batch
            this.IsDone = false;
            this.MiniBatchSize = this.InitialMiniBatchSize;
            if ~isempty(this.Datasource)
                this.Datasource.reset();
            end
            this.CurrentStartIndex = 1;
        end
        
        function shuffle(this)
            % shuffle   Shuffle the data
            this.Datasource.shuffle();
        end
        
        function reorder(this,indices)
           % reorder  Shuffle the data in a specific order
           %
           % Note, this will only be called if underlying Datasource implements
           % reorder, as checked in implementsReorder().
           
           this.checkValidReorderIndices(indices);
           this.Datasource.reorder(indices);
        end
        
        function set.MiniBatchSize(this, value)
            if ~isempty(this.Datasource)
                value = min(value, this.NumObservations);
                this.Datasource.MiniBatchSize = value;
            end
        end
        
        function batchSize = get.MiniBatchSize(this)
            if isempty(this.Datasource)
                batchSize = 0;
            else
                batchSize = this.Datasource.MiniBatchSize;
            end
        end
        
        function TF = implementsReorder(this)
            % implementsReorder  Whether a dispatcher implements the reorder interface.
            % In this particular dispatcher, we need to interrogate the underlying MiniBatchDatasource.
            % This is an overload of the definition in
            % BackgroundCapableDispatcher.
            
            TF = ismethod(this.Datasource,'reorder');
            
        end
        
    end
    
    methods (Access = protected)
        

        
    end
    
    methods (Access = private)
        
        function [miniBatchData, miniBatchResponse] = readData(this)
            % readData  Read data and response corresponding to indices
            [X,Y] = this.Datasource.nextBatch();
            miniBatchData = this.readObservations(X);
            miniBatchResponse = this.readResponses(Y);
        end
        
        function observations = readObservations(this,X)
            if isempty(X)
                observations = [];
            else
                observations = iCellTo4DArray(X);
            end
            observations = this.Precision.cast(observations);
        end
        
        function responses = readResponses(this,Y)
            if isempty(Y)
                responses = [];
            else
                if iscell(Y) % Used by 3-D array Table syntax
                    responses = iCellTo4DArray(Y);
                elseif iscategorical(Y)
                    % Categorical vector of responses
                    responses = iDummify( Y );                
                else
                    responses = Y;
                end
            end
            % Cast to the right precision
            responses = this.Precision.cast( responses );
        end
        
    end
    
    methods (Access = private)
        
        function [exampleInput,exampleResponse,classNames] = getExampleInfoFromDatasource(this)
            % getExampleInfoFromDatasource   Extract examples of input,
            % response, and classnames from a datasource.
            
            this.MiniBatchSize = 1;
            [X,Y] = nextBatch(this.Datasource);
            
            exampleInput = this.readObservations(X);
            this.MiniBatchSize = this.InitialMiniBatchSize;
            exampleResponse = this.readResponses(Y);
            
            classNames = iGetClassNames(Y);
            
            this.Datasource.reset();
        end
        
        function indices = currentIndices(this)
            indices = this.CurrentStartIndex : (this.CurrentStartIndex + this.MiniBatchSize - 1);
        end
        
        function [oldIdx, newIdx] = advanceCurrentStartIndex( this, n )
            % advanceCurrentStartIndex     Advance current index of n positions and return
            % its old and new value
            oldIdx = this.CurrentStartIndex;
            this.CurrentStartIndex = this.CurrentStartIndex + n;
            newIdx = this.CurrentStartIndex;
        end
        
        function miniBatchSize = nextMiniBatchSize( this, currentEndIdx )
            % nextMiniBatchSize   Compute the size of the next mini batch
                        
            miniBatchSize = min( this.MiniBatchSize, this.NumObservations - currentEndIdx );
            
            if isequal(this.EndOfEpoch, 'discardLast') && miniBatchSize<this.MiniBatchSize
                miniBatchSize = 0;
            end
        end
        
    end
    
end

function data = iCellTo4DArray( images )
% iCellTo4DArray   Convert a cell array of images to a 4-D array. If the
% input images is already an array just return it.
if iscell( images )
    try
        data = cat(4, images{:});
    catch e
        throwVariableSizesException(e);
    end
else
    data = images;
end
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(errorID, getString(message(errorID, varargin{:})));
end

function throwVariableSizesException(e)
% throwVariableSizesException   Throws a subsassigndimmismatch exception as
% a VariableImageSizes exception
if (strcmp(e.identifier,'MATLAB:catenate:dimensionMismatch'))
    exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:MiniBatchDatasourceDispatcher:VariableImageSizes');
    throwAsCaller(exception)
else
    rethrow(e)
end
end

function dummy = iDummify(categoricalIn)
% iDummify   Dummify a categorical vector of size numObservations x 1 to
% return a 4-D array of size 1 x 1 x numClasses x numObservations
if isempty( categoricalIn )
    numClasses = 0;
    numObs = 1;
    dummy = zeros( 1, 1, numClasses, numObs );
else
    if isvector(categoricalIn)
        dummy = nnet.internal.cnn.util.dummify(categoricalIn);
    else
        dummy = iDummify4dArray(categoricalIn);
    end
end
end

function dummy = iDummify4dArray(C)
numCategories = numel(categories(C));
[H, W, ~, numObservations] = size(C);
dummifiedSize = [H, W, numCategories, numObservations];
dummy = zeros(dummifiedSize, 'single');
C = iMakeVertical( C );

[X,Y,Z] = meshgrid(1:W, 1:H, 1:numObservations);

X = iMakeVertical(X);
Y = iMakeVertical(Y);
Z = iMakeVertical(Z);

% Remove missing labels. These are pixels we should ignore during
% training. The dummified output is all zeros along the 3rd dims and are
% ignored during the loss computation.
[C, removed] = rmmissing(C);
X(removed) = [];
Y(removed) = [];
Z(removed) = [];

idx = sub2ind(dummifiedSize, Y(:), X(:), int32(C), Z(:));
dummy(idx) = 1;
end

function classnames = iGetClassNames(response)
if isa(response, 'categorical')
    classnames = categories( response );
else
    classnames = {};
end
end

function imageSize = iImageSize(image)
% iImageSize    Retrieve the image size of an image, adding a third
% dimension if grayscale
[ imageSize{1:3} ] = size(image);
imageSize = cell2mat( imageSize );
end

function responseSize = iResponseSize(response)
% iResponseSize   Return the size of the response in the first three
% dimensions.
[responseSize(1), responseSize(2), responseSize(3), ~] = size(response);
end

function vec = iMakeVertical( vec )
    vec = reshape( vec, numel( vec ), 1 );
end