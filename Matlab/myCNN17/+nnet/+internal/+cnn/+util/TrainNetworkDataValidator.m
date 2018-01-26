classdef (Sealed) TrainNetworkDataValidator
    % TrainNetworkDataValidator   Class that holds various validation
    % functions for trainNetwork data
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Access = private)
        % ErrorStrategy (nnet.internal.cnn.util.ErrorStrategy)   Error strategy
        ErrorStrategy
    end
    
    methods
        function this = TrainNetworkDataValidator( errorStrategy )
            this.ErrorStrategy = errorStrategy;
        end
        
        function validateDataForProblem( this, X, Y, layers )
            % validateDataForProblem   Assert that the input data X and
            % response Y are valid for the class of problem considered
            internalLayers = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers( layers );
            isAClassificationNetwork = iIsInternalClassificationNetwork(internalLayers);
            isRNN = nnet.internal.cnn.util.isRNN( internalLayers );
            if iIsAnImageDatastore( X ) && ~isRNN
                this.assertValidInputImageDatastore( X );
                this.assertClassificationForImageDatastore(isAClassificationNetwork);
            elseif istable( X )
                this.assertValidTable( X, isAClassificationNetwork );
            elseif isnumeric( X ) && ~isRNN
                this.assertValidImageArray( X );
                this.assertValidResponseForTheNetwork( Y, isAClassificationNetwork );
                this.assertXAndYHaveSameNumberOfObservations( X, Y );
            elseif isRNN
                this.assertNotDatastoreOrSource( X );
                this.assertValidSequenceInput( X );
                this.assertValidSequenceResponse( Y, isAClassificationNetwork );
                this.assertSequencesHaveSameNumberOfObservations( X, Y );
                this.assertOutputModeCorrespondsToData( X, Y, internalLayers );
                this.assertResponsesHaveValidSequenceLength( X, Y );
            elseif iIsADataDispatcher( X )
                % X is a custom dispatcher - the custom dispatcher api is
                % for internal use only
            elseif iIsAMiniBatchDatasource( X )
                
            else
                this.ErrorStrategy.throwXIsNotValidType();
            end
        end
        
        function validateDataSizeForNetwork( this, dispatcher, internalLayers )
            % validateDataSizeForNetwork   Assert that the data to be
            % dispatched by dispatcher is of size that fits the network
            % represented by internalLayers
            this.assertCorrectDataSizeForInputLayer(dispatcher, internalLayers{1})
            this.assertCorrectResponseSizeForOutputLayer(dispatcher, internalLayers)
        end
        
        function validateDataSizeForDAGNetwork( this, dispatcher, lgraph )
            % validateDataSizeForNetwork   Assert that the data to be
            % dispatched by dispatcher is of size that fits the network
            % represented by lgraph
            internalLayers = iGetInternalLayers(lgraph.Layers);
            this.assertCorrectDataSizeForInputLayer(dispatcher, internalLayers{1})
            this.assertCorrectResponseSizeForDAGOutputLayer(dispatcher, lgraph, internalLayers)
        end
    end
    
    methods (Access = private)
        %% Assert helpers
        function assertLabelsAreDefined(this, labels)
            if iscategorical( labels )
                if(any(isundefined(labels)))
                    this.ErrorStrategy.throwUndefinedLabels()
                end
            else 
                % Assume filepath labels. Load the first response and check
                % for undefined labels. Loading all responses may be
                % time-consuming, so we load only the first
                D = load( labels{1} );
                response = iReadDataFromStruct( D );
                if(any(isundefined(response)))
                    this.ErrorStrategy.throwUndefinedLabels()
                end
            end
        end
        
        function assertValidImageArray(this, x)
            if isa(x, 'gpuArray') || ~isnumeric(x) || ~isreal(x) || ~iIsValidImageArray(x)
                this.ErrorStrategy.throwXIsNotValidImageArray()
            end
        end
        
        function assertValidRegressionResponse(this, x)
            if isa(x, 'gpuArray') || ~isnumeric(x) || ~isreal(x) || ~iIsValidResponseArray(x)
                this.ErrorStrategy.throwYIsNotValidResponseArray()
            end
        end
        
        function assertValidResponseForTheNetwork(this, x, isAClassificationNetwork)
            % assertValidResponseForTheNetwork   Assert if x is a valid response for
            % the type of network in use.
            if isAClassificationNetwork
                this.assertCategoricalResponseVector(x);
                this.assertLabelsAreDefined(x);
            else
                this.assertValidRegressionResponse(x);
            end
        end
        
        function assertCategoricalResponseVector(this, x)
            if ~(iscategorical(x) && isvector(x))
                this.ErrorStrategy.throwYIsNotCategoricalResponseVector()
            end
        end
        
        function assertXAndYHaveSameNumberOfObservations(this, x, y)
            if size(x,4)~=iArrayResponseNumObservations(y)
                this.ErrorStrategy.throwXAndYHaveDifferentObservations()
            end
        end
        
        function assertClassificationForImageDatastore(this, isAClassificationNetwork)
            if ~isAClassificationNetwork
                this.ErrorStrategy.throwImageDatastoreWithRegression()
            end
        end
        
        function assertValidTable(this, tbl, isAClassificationNetwork)
            % assertValidTable   Assert that tbl is a valid table according to the
            % type of network defined by layers (classification or regression).
            if isAClassificationNetwork
                this.assertValidClassificationTable(tbl);
            else
                this.assertValidRegressionTable(tbl);
            end
        end
        
        function assertValidClassificationTable(this, tbl)
            % assertValidClassificationTable   Assert that tbl is a valid
            % classification table. To be valid, it needs to have image paths or images
            % in the first column. Responses will be held in the second column as
            % categorical labels.
            isValidFirstColumn = iHasValidPredictorColumn(tbl);
            hasValidResponses = iHasValidClassificationResponses(tbl);
            isValidClassificationTable = isValidFirstColumn && hasValidResponses;
            if ~isValidClassificationTable
                this.ErrorStrategy.throwInvalidClassificationTable()
            end
            % Assert that all labels are defined for the classification problem
            this.assertLabelsAreDefined(tbl{:,2});
        end
        
        function assertValidRegressionTable(this, tbl)
            % assertValidRegressionTable   Assert that tbl is a valid regression
            % table. To be valid, it needs to have image paths or images in the first
            % column. Responses will be held in the second column as either vectors or
            % cell arrays containing 3-D arrays. Alternatively, responses will be held
            % in multiple columns as scalars.
            if ~iHasValidPredictorColumn(tbl)
                this.ErrorStrategy.throwInvalidRegressionTablePredictors()
            end
            if  ~iHasValidRegressionResponses(tbl)
                this.ErrorStrategy.throwInvalidRegressionTableResponses()
            end
        end
        
        function assertValidInputImageDatastore(this, X)
            % assertValidInputImageDatastore   Assert that X is a valid image
            % datastore to be used in trainNetwork
            this.assertValidImageDatastore( X );
            this.assertDatastoreHasLabels( X );
            this.assertDatastoreLabelsAreCategorical( X );
            this.assertLabelsAreDefined( X.Labels );
        end
        
        function assertValidImageDatastore(this, imds)
            if ~iIsAnImageDatastore(imds)
                this.ErrorStrategy.throwNotAnImageDatastore()
            end
        end
        
        function assertDatastoreHasLabels(this, imds)
            if isempty(imds.Labels)
                this.ErrorStrategy.throwImageDatastoreHasNoLabels()
            end
        end
        
        function assertDatastoreLabelsAreCategorical(this, imds)
            if ~iscategorical(imds.Labels)
                this.ErrorStrategy.throwImageDatastoreMustHaveCategoricalLabels()
            end
        end
        
        function assertCorrectDataSizeForInputLayer(this, dispatcher, internalInputLayer)
            if isa( internalInputLayer, 'nnet.internal.cnn.layer.ImageInput' )
                if(~internalInputLayer.isValidTrainingImageSize(dispatcher.ImageSize))
                    this.ErrorStrategy.throwImagesInvalidSize( ...
                        i3DSizeToString(dispatcher.ImageSize), ...
                        i3DSizeToString(internalInputLayer.InputSize) );
                end
            elseif isa( internalInputLayer, 'nnet.internal.cnn.layer.SequenceInput' )
                if(~internalInputLayer.isValidInputSize(dispatcher.DataSize))
                    this.ErrorStrategy.throwSequencesInvalidSize( ...
                        int2str(dispatcher.DataSize), ...
                        int2str(internalInputLayer.InputSize) );
                end
            end
        end
        
        function assertCorrectResponseSizeForOutputLayer(this, dispatcher, internalLayers)
            networkOutputSize = iNetworkOutputSize(internalLayers);
            dataResponseSize = dispatcher.ResponseSize;
            
            if ~isequal( networkOutputSize, dataResponseSize )
                if iIsInternalClassificationNetwork(internalLayers)
                    networkClasses = networkOutputSize(end);
                    expectedClasses = dataResponseSize(end);
                    this.ErrorStrategy.throwOutputSizeNumClassesMismatch( ...
                        mat2str( networkClasses ), mat2str( expectedClasses ) );
                else
                    if iIsScalarResponseSize( networkOutputSize ) && iIsScalarResponseSize( dataResponseSize )
                        % If both responses are scalar, output a different error
                        % message regarding number of responses
                        networkResponses = networkOutputSize(3);
                        dataResponses = dataResponseSize(3);
                        this.ErrorStrategy.throwOutputSizeNumResponsesMismatch( ...
                            mat2str( networkResponses ), mat2str( dataResponses ) );
                    else
                        this.ErrorStrategy.throwOutputSizeResponseSizeMismatch( ...
                            mat2str( networkOutputSize ), mat2str( dataResponseSize ) );
                    end
                end
            end
        end
        
        function assertCorrectResponseSizeForDAGOutputLayer(this, dispatcher, lgraph, internalLayers)
            networkOutputSize = iDAGNetworkOutputSize(lgraph);
            dataResponseSize = dispatcher.ResponseSize;
            
            if ~isequal( networkOutputSize, dataResponseSize )
                if iIsInternalClassificationNetwork(internalLayers)
                    networkClasses = networkOutputSize(3);
                    expectedClasses = dataResponseSize(3);
                    this.ErrorStrategy.throwOutputSizeNumClassesMismatch( ...
                        mat2str( networkClasses ), mat2str( expectedClasses ) );
                else
                    if iIsScalarResponseSize( networkOutputSize ) && iIsScalarResponseSize( dataResponseSize )
                        % If both responses are scalar, output a different error
                        % message regarding number of responses
                        networkResponses = networkOutputSize(3);
                        dataResponses = dataResponseSize(3);
                        this.ErrorStrategy.throwOutputSizeNumResponsesMismatch( ...
                            mat2str( networkResponses ), mat2str( dataResponses ) );
                    else
                        this.ErrorStrategy.throwOutputSizeResponseSizeMismatch( ...
                            mat2str( networkOutputSize ), mat2str( dataResponseSize ) );
                    end
                end
            end
        end
        
        function assertNotDatastoreOrSource(this, x)
            if iIsAnImageDatastore(x) || iIsAMiniBatchDatasource(x)
                this.ErrorStrategy.throwIncompatibleInputForLSTM()
            end
        end
        
        function assertValidSequenceInput(this, x)
            if ~iIsValidSequenceInput(x)
                this.ErrorStrategy.throwXIsNotValidSequenceInput()
            end
            if ~iSequencesHaveConsistentDataDimension(x)
                this.ErrorStrategy.throwXIsNotValidSequenceInput()
            end
        end
        
        function assertValidSequenceResponse(this, x, isAClassificationNetwork)
            if isAClassificationNetwork
                if ~iIsValidSequenceCategoricalResponse(x)
                    this.ErrorStrategy.throwYIsNotValidSequenceCategorical()
                end
                if ~iHasDefinedSequenceLabels(x)
                    this.ErrorStrategy.throwUndefinedLabels()
                end
            end
        end
        
        function assertSequencesHaveSameNumberOfObservations(this, x, y)
            if ~iSequencesHaveSameNumberOfObservations(x, y)
                this.ErrorStrategy.throwXAndYHaveDifferentObservations()
            end
        end
        
        function assertResponsesHaveValidSequenceLength(this, x, y)
            if ~iSequencesHaveConsistentSequenceLength(x, y)
                this.ErrorStrategy.throwInvalidResponseSequenceLength()
            end
        end
        
        function assertOutputModeCorrespondsToData(this, x, y, internalLayers)
            returnsSequence = nnet.internal.cnn.util.returnsSequence( internalLayers, true );
            if ~iOutputModeMatchesData(returnsSequence, x, y)
                if returnsSequence
                    this.ErrorStrategy.throwOutputModeSequenceDataMismatch()
                else
                    this.ErrorStrategy.throwOutputModeLastDataMismatch()
                end
            end
        end
        
    end
end

%% ISA/HASA helpers
function tf = iIsInternalClassificationNetwork(internalLayers)
tf = iIsInternalClassificationLayer( internalLayers{end} );
end

function tf = iIsInternalClassificationLayer(internalLayer)
tf = isa(internalLayer, 'nnet.internal.cnn.layer.ClassificationLayer');
end

function tf = iIsAnImageDatastore(x)
tf = isa(x, 'matlab.io.datastore.ImageDatastore');
end

function tf = iIsADataDispatcher(X)
tf = isa(X, 'nnet.internal.cnn.DataDispatcher');
end

function tf = iIsAMiniBatchDatasource(X)
tf = isa(X, 'nnet.internal.cnn.MiniBatchDatasource');
end

function tf = iIsValidImageArray(x)
% iIsValidImageArray   Return true if x is an array of
% one or multiple (colour or grayscale) images
tf = iIsRealNumericData( x ) && ...
    ( iIsGrayscale( x ) || iIsColour( x ) ) && ...
    iIs4DArray( x );
end

function tf = iIsValidImage(x)
% iIsValidImage   Return true if x is a non-empty (colour or grayscale) image
tf = ~isempty( x ) && iIsRealNumericData( x ) && ...
    ( iIsGrayscale( x ) || iIsColour( x ) ) && ...
    ndims( x ) < 4 ;
end

function tf = iIsValidPath(x)
% iIsValidPath   Return true if x is a valid path. For the moment, we just
% assume any char vector is a valid path.
tf = ischar(x);
end

function tf = iIsValidResponseArray(x)
% iIsValidResponseArray   Return true if x is a vector, a matrix or an
% array of real responses and it does not contain NaNs.
tf = iIsRealNumericData( x ) && ...
    ( isvector(x) || ismatrix(x) || iIs4DArray( x ) ) && ...
    ~iContainsNaNs( x );
end

function tf = iIsRealNumericData(x)
tf = isreal(x) && isnumeric(x);
end

function tf = iIsRealNumericVector(x)
tf = iIsRealNumericData(x) && isvector(x);
end

function tf = iContainsNaNs(x)
if isnumeric(x)
    tf = any(isnan(x(:)));
else
    % If x is not numeric, it cannot contain NaNs since NaN is numeric
    tf = false;
end
end

function tf = iIsGrayscale(x)
tf = size(x,3)==1;
end

function tf = iIsColour(x)
tf = size(x,3)==3;
end

function tf = iIs3DArray(x)
tf = ndims( x ) <= 3;
end

function tf = iIs4DArray(x)
tf = ndims( x ) <= 4;
end

function tf = iHasValidPredictorColumn(tbl)
% iHasValidPredictorColumn   Return true if tbl has a valid predictor
% column as first column. To be valid, the first column should be a cell
% array containing only paths or only image data.
tf = iFirstColumnIsCell(tbl) && ...
    ( iFirstColumnContainsOnlyPaths(tbl) || iFirstColumnContainsOnlyImages(tbl) );
end

function tf = iFirstColumnIsCell(tbl)
% iFirstColumnIsCell   Return true if the first column of tbl is a cell
% array.
tf = iscell(tbl{:,1});
end

function tf = iFirstColumnContainsOnlyPaths(tbl)
% iFirstColumnContainsOnlyPaths   Return true if the first column of tbl
% contains paths and only paths. We do not check if those paths exist,
% since that might be too time consuming.
res = cellfun(@iIsValidPath, tbl{:,1});
tf = all(res);
end

function tf = iFirstColumnContainsOnlyImages(tbl)
% iFirstColumnContainsOnlyImages   Return true if the first column of tbl
% contains images.
res = cellfun(@iIsValidImage, tbl{:,1});
tf = all(res);
end

function tf = iSecondColumnContainsOnlyPaths(tbl)
% iSecondColumnContainsOnlyPaths   Return true if the second column of tbl
% contains paths and only paths. We do not check if those paths exist,
% since that might be too time consuming.
res = cellfun(@iIsValidPath, tbl{:,2});
tf = all(res);
end

function tf = iIsACellOf3DNumericArray(x)
tf = iscell(x) && isnumeric(x{:}) && iIs3DArray(x{:});
end

function tf = iHasValidClassificationResponses(tbl)
% iHasValidClassificationResponses   Return true if tbl contains only one
% response column (the second), and responses are stored as a categorical
% vector.
numResponses = size(tbl,2) - 1;
if numResponses ~= 1
    % The table has an incorrect number of response columns
    tf = false;
elseif iscell( tbl{:,2} ) 
    % The table contains file paths of categorical predictors
    tf = iSecondColumnContainsOnlyPaths(tbl);
else
    responses = tbl{:,2};
    responsesAreCategorical = iscategorical(responses);
    % We check the size of the second dimension instead of using isvector
    % to avoid considering horizontal vectors in case there is only one
    % response
    isAVectorOfResponses = size(responses,2)==1;
    tf = isAVectorOfResponses && responsesAreCategorical;
end
end

function tf = iHasValidRegressionResponses(tbl)
% iHasValidRegressionResponses   Return true if tbl contains real scalar
% responses in all columns except the first, or if responses are held in
% one column only in the form of either vectors or cell arrays containing
% 3-D arrays. Responses cannot contain NaNs.
numResponses = width(tbl) - 1;
if numResponses < 1
    % The table has no response column
    tf = false;
elseif numResponses == 1
    % The table has one response column: responses can be scalars, vectors
    % or cell containing 3-D arrays
    scalarOrVectorResponses = all( rowfun(@(x)iIsRealNumericVector(x), tbl(:,2), 'OutputFormat', 'uniform') );
    cellOf3DArrayResponses = all( rowfun(@(x)iIsACellOf3DNumericArray(x), tbl(:,2), 'OutputFormat', 'uniform') );
    containsNaNs = iColumnContainsNaNs(tbl{:,2});
    hasEmptyCells = iColumnContainsEmptyValues(tbl{:,2});
    tf = scalarOrVectorResponses || cellOf3DArrayResponses;
    tf = tf && ~containsNaNs && ~hasEmptyCells;
else
    % There are multiple response columns: responses can only be scalars
    isRealNumericColumnWithoutNaNsFcn = @(x)(iIsRealNumericVector(x) && ~iColumnContainsNaNs(x));
    tfOnEachColumn = varfun(isRealNumericColumnWithoutNaNsFcn, tbl(:,2:end), 'OutputFormat', 'uniform');
    tf = all( tfOnEachColumn );
end
end

function tf = iColumnContainsNaNs( x )
if iscell( x )
    tf = any(cellfun(@iContainsNaNs, x));
else
    tf = iContainsNaNs( x );
end
end

function tf = iColumnContainsEmptyValues( x )
if iscell( x )
    tf = any(cellfun(@isempty, x));
else
    tf = any(isempty(x));
end
end

function tf = iIsScalarResponseSize( x )
% iIsScalarResponse   Return true if the first two dimensions of the 3-D
% response size x are ones.
tf = x(1)==1 && x(2)==1;
end

function tf = iIsValidSequenceInput( x )
% iIsValidSequenceInput   Return true for cell arrays with real, numeric
% entries, or a single sequence represented by a numeric matrix
validCell = iscell(x) && isvector(x) && all( cellfun(@(s)isreal(s), x) );
validMatrix = isnumeric(x) && ismatrix(x);
tf = validCell || validMatrix;
end

function tf = iIsValidSequenceCategoricalResponse( x )
% iIsValidSequenceCategoricalResponse   Return true for valid categorical
% or cell array categorical responses
validSeq2OneResponse = iscategorical(x) && isvector(x);
validSeq2SeqResponse = iscell(x) && isvector(x) && all( cellfun(@(s)iscategorical(s) && isrow(s), x) ) ...
    || (iscategorical(x) && isrow(x));
tf = validSeq2OneResponse || validSeq2SeqResponse;
end

function tf = iHasDefinedSequenceLabels( x )
% iHasDefinedSequenceLabels   Return true if sequence response has no
% undefined labels
validSeq2OneResponse = iscategorical(x) && all( ~isundefined(x) );
validSeq2SeqResponse = iscell(x) && all( cellfun(@(s)all( ~isundefined(s) ), x) );
tf = validSeq2OneResponse || validSeq2SeqResponse;
end

function tf = iSequencesHaveConsistentDataDimension( x )
% iSequencesHaveConsistentDataDimension   Return true if all sequences
% within a cell array have the same data dimension
if iscell(x)
    firstSize = size( x{1}, 1 );
    tf = all( cellfun( @(s)isequal( size(s, 1), firstSize ), x ) );
else
    % If the sequences are not in a cell, then there is only one
    % observation and dimensions are consistent by default
    tf = true;
end
end

function tf = iSequencesHaveSameNumberOfObservations(x, y)
% iSequencesHaveSameNumberOfObservations   Return true if sequence
% predictors and responses have the same number of observations
if iscell(x)
    tf = numel(x) == numel(y);
else
    % If the predictors are not in a cell, then there should only be one
    % observation
    tf = ismatrix(x) && iscategorical(y);
end
end

function tf = iSequencesHaveConsistentSequenceLength( x, y )
% iSequencesHaveConsistentSequenceLength   Return true if sequence
% responses have the same number of timesteps as the corresponding
% predictors
if iscell(y)
    sx = cellfun( @(s)size(s, 2), x );
    sy = cellfun( @(s)size(s, 2), y );
    tf = all( sx == sy );
elseif isnumeric(x) && iscategorical(y)
    % Assume single observation seq-to-seq case
    tf = size(x, 2) == size(y, 2);
else
    % Otherwise sequences are just labels, so then this is true by default 
    tf = true;
end
end

function tf = iOutputModeMatchesData(returnsSequence, x, y)
validSeq2Seq = returnsSequence && ((iscell(x) && iscell(y)) || (isnumeric(x) && iscategorical(y)));
validSeq2One = ~returnsSequence && (iscell(x) && iscategorical(y) && iscolumn(y));
tf = validSeq2One || validSeq2Seq;
end

%% Generic helpers
function arraySize = iArrayResponseNumObservations(y)
% iArrayResponseNumObservations   Return the number of observations of the
% response array y. The number of observations will be the number of
% elements for a vector, the first dimension for a matrix and the last
% dimension when the responses are stored in a 4-D array.
if isvector( y )
    arraySize = numel( y );
elseif ismatrix( y )
    arraySize = size( y, 1);
else
    arraySize = size( y, 4 );
end
end

function sizeString = i3DSizeToString( sizeVector )
% i3DSizeToString   Convert a 3-D size stored in a vector of 3 elements
% into a string separated by 'x'.
sizeString = [ ...
    int2str( sizeVector(1) ) ...
    'x' ...
    int2str( sizeVector(2) ) ...
    'x' ...
    int2str( sizeVector(3) ) ];
end

function outputSize = iNetworkOutputSize(internalLayers)
% Determine the output size of the network given the internal layers
inputSize = internalLayers{1}.InputSize;
for i = 2:numel(internalLayers)
    inputSize = internalLayers{i}.forwardPropagateSize(inputSize);
end
outputSize = inputSize;
end

function outputSize = iDAGNetworkOutputSize(lgraph)
% Determine the output size of the network given the layer graph
sizes = extractSizes(lgraph);
outputSize = sizes{end};
end

function internalLayers = iGetInternalLayers( layers )
internalLayers = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers( layers );
end

function data = iReadDataFromStruct(S)
% Read data from first field in struct S
fn = fieldnames( S );
data = S.(fn{1});
end