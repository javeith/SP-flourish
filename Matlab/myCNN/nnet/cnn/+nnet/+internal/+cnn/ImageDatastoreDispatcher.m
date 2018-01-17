classdef ImageDatastoreDispatcher < nnet.internal.cnn.DataDispatcher
    % ImageDatastoreDispatcher  class to dispatch 4D data from
    %   ImageDatastore data
    %
    % Input data    - an image datastore containing either RGB or grayscale
    %               images of the same size
    % Output data   - 4D data where the fourth dimension is the number of
    %               observations in that mini batch. If the input is a
    %               grayscale image then the 3rd dimension will be 1
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % ImageSize  (1x3 int) Size of each image to be dispatched
        ImageSize
        
        % NumObservations (int) Number of observations in the data set
        NumObservations
        
        % ClassNames (cellstr) Array of class names corresponding to
        %            training data labels.
        ClassNames
        
        % PrivateMiniBatchSize (int)   Number of elements in a mini batch
        PrivateMiniBatchSize
    end
    
    properties(SetAccess = public)
        % EndOfEpoch    Strategy for how to cope with the last mini-batch when the number
        % of observations is not divisible by the number of mini batches.
        %
        % Allowed values: 'truncateLast', 'discardLast'
        EndOfEpoch
        
        % Precision Precision used for dispatched data
        Precision
    end
    
    properties (SetAccess = private, Dependent)
        % IsDone (logical)     True if there is no more data to dispatch
        IsDone
    end
    
    properties (Access = private)
        % Datastore  (ImageDatastore)     The ImageDatastore we are going to
        % read data and responses from
        Datastore
        
        % CurrentIndex  (int)   Current index of image to be dispatched
        CurrentIndex
        
        % ExampleImage    An example of the images in the dataset
        ExampleImage
        
        % ExampleLabel     An example categorical label from the dataset
        ExampleLabel
    end
    
    properties(Dependent)
        % MiniBatchSize (int)   Number of elements in a mini batch
        MiniBatchSize
    end
    
    methods
        function this = ImageDatastoreDispatcher(imageDatastore, miniBatchSize, endOfEpoch, precision)
            % ImageDatastoreDispatcher   Constructor for array data dispatcher
            %
            % imageDatastore    - An ImageDatastore containing the images
            % miniBatchSize     - Size of a mini batch express in number of
            %                   images
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
            
            this.Datastore = imageDatastore;
            % Count the number of observations
            this.NumObservations = numel( imageDatastore.Files );
            
            this.MiniBatchSize = miniBatchSize;
            
            assert(isequal(endOfEpoch,'truncateLast') || isequal(endOfEpoch,'discardLast'), 'nnet.internal.cnn.ImageDatastoreDispatcher error: endOfEpoch should be one of ''truncateLast'', ''discardLast''.');
            this.EndOfEpoch = endOfEpoch;
            this.Precision = precision;
            
            % Read an example image
            [this.ExampleImage, exampleInfo] = imageDatastore.readimage(1);
            
            % Record an example label to capture type
            this.ExampleLabel = exampleInfo.Label;
            
            % Set the expected image size
            this.ImageSize = iImageSize( this.ExampleImage );
            
            % Get class names from labels
            this.ClassNames = iGetClassNames(this.Datastore);
            
        end
        
        function tf = get.IsDone( this )
            if isequal(this.EndOfEpoch, 'truncateLast')
                tf = this.CurrentIndex >= this.NumObservations;
            else % discardLast
                % If the current index plus the mini-batch size takes us
                % beyond the end of the number of observations, then we are
                % done
                tf = this.CurrentIndex + this.MiniBatchSize > this.NumObservations;
            end
        end
        
        function [miniBatchData, miniBatchResponse, miniBatchIndices] = next(this)
            % next   Get the data and response for the next mini batch and
            % correspondent indices
            currentMiniBatchSize = this.nextMiniBatchSize();
            if currentMiniBatchSize>0
                miniBatchLabels = repmat( this.ExampleLabel, currentMiniBatchSize, 1 );
                this.Datastore.ReadSize = currentMiniBatchSize;
                [images, info] = this.Datastore.read();
                
                miniBatchData = iCellTo4DArray( images );
                
                % If there are no categories don't try to record empty
                % labels
                if ~isempty(info.Label)
                    miniBatchLabels = info.Label;
                end
                miniBatchIndices = this.nextIndices( currentMiniBatchSize );
                
                % Convert to correct types
                miniBatchData = this.Precision.cast( miniBatchData );
                miniBatchResponse = this.Precision.cast( iDummify( miniBatchLabels, currentMiniBatchSize ) );
            else
                miniBatchData = [];
                miniBatchResponse = [];
                miniBatchIndices = [];
            end
        end
        
        function start(this)
            % start     Set the next the mini batch to be the first mini
            % batch in the epoch
            reset( this.Datastore );
            this.CurrentIndex = 0;
        end
        
        function shuffle(this)
            % shuffle   Shuffle the data
            this.Datastore = this.Datastore.shuffle();
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
        function indices = nextIndices( this, n )
            % nextIndices   Return the next n indices
            [startIdx, endIdx] = this.advanceCurrentIndex( n );
            indices = startIdx+1:endIdx;
            % The returned indices are expected to be a column vector
            indices = indices';
        end
        
        function [oldIdx, newIdx] = advanceCurrentIndex( this, n )
            % nextIndex     Advance current index of n positions and return
            % its old and new value
            oldIdx = this.CurrentIndex;
            this.CurrentIndex = this.CurrentIndex + n;
            newIdx = this.CurrentIndex;
        end
        
        function miniBatchSize = nextMiniBatchSize( this )
            % nextMiniBatchSize   Compute the size of the next mini batch
            miniBatchSize = min( this.MiniBatchSize, this.NumObservations - this.CurrentIndex );
            
            if isequal(this.EndOfEpoch, 'discardLast') && miniBatchSize<this.MiniBatchSize
                miniBatchSize = 0;
            end
        end
    end
end

function imageSize = iImageSize(image)
% iImageSize    Retrieve the image size of an image, adding a third
% dimension if grayscale
imageSize = size(image);
% If image is grayscale, add another dimension
if numel(imageSize)==2
    imageSize = [imageSize 1];
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

function classnames = iGetClassNames(imds)
if isa(imds.Labels, 'categorical')
    classnames = categories( imds.Labels );
else
    classnames = {};
end
end

function dummy = iDummify(categoricalIn, numObservations)
if isempty(categoricalIn)
    numClasses = 0;
    dummy = zeros(1, 1, numClasses, numObservations);
else
    dummy = nnet.internal.cnn.util.dummify(categoricalIn);
end
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(errorID, getString(message(errorID, varargin{:})));
end

function throwVariableSizesException(e)
% throwVariableSizesException   Throws a subsassigndimmismatch exception as
% a VariableImageSizes exception
if (strcmp(e.identifier,'MATLAB:catenate:dimensionMismatch'))
    exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:ImageDatastoreDispatcher:VariableImageSizes');
    throwAsCaller(exception)
else
    rethrow(e)
end
end
