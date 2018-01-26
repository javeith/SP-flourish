classdef DataDispatcherFactory
    % DataDispatcherFactory   Factory for making data dispatchers
    %
    %   dataDispatcher = DataDispatcherFactoryInstance.createDataDispatcher(data, options)
    %   data: the data to be dispatched.
    %       According to their type the appropriate dispatcher will be used.
    %       Supported types: 4-D double, imagedatastore, table
    %   options: input arguments for the data dispatcher (e.g. response vector,
    %   mini batch size)
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    methods (Static)
        function dispatcher = createDataDispatcher( inputs, response, ...
                miniBatchSize, endOfEpoch, precision, executionSettings, ...
                shuffleOption, sequenceLength, paddingValue, layers )
            % createDataDispatcher   Create data dispatcher
            %
            % Syntax:
            %     createDataDispatcher( inputs, response, ... 
            % miniBatchSize, endOfEpoch, precision, executionSettings, ...
            % shuffleOption, sequenceLength, paddingValue, layers )
            
            % Allow executionSettings to be unspecified
            if nargin < 6
                executionSettings = struct( 'useParallel', false );
            end
            % Allow shuffle setting to be unspecified
            if nargin < 7
                shuffleOption = 'once';
            end
            % Allow sequenceLength to be unspecified
            if nargin < 8
                sequenceLength = 'longest';
            end
            % Allow paddingValue to be unspecified
            if nargin < 9
                paddingValue = 0;
            end
            % Allow layers to be unspecified
            if nargin < 10
                layers = nnet.cnn.layer.Layer.empty();
            end
            
            if isa(inputs, 'nnet.internal.cnn.DataDispatcher')
                dispatcher = inputs;
                
                % Setup the dispatcher to factory specifications.
                dispatcher.EndOfEpoch    = endOfEpoch;
                dispatcher.Precision     = precision;
                dispatcher.MiniBatchSize = miniBatchSize; 
            else
                isClassificationRNN = iIsSequenceClassificationLayers(layers);
                if iIsRealNumeric4DHostArray(inputs) && ~isClassificationRNN
                    datasource = iCreate4dArrayMiniBatchDatasource( inputs, response, miniBatchSize );
                elseif isa(inputs, 'matlab.io.datastore.ImageDatastore')
                    datasource  = iCreateImageDatastoreMiniBatchDatasource( inputs, miniBatchSize );
                elseif isa(inputs, 'nnet.internal.cnn.MiniBatchDatasource')
                    datasource = inputs;
                elseif istable(inputs) && ~isClassificationRNN
                    if iIsAnInMemoryTable(inputs)
                        datasource = iCreateInMemoryTableMiniBatchDatasource( inputs, miniBatchSize );
                    else
                        datasource  = iCreateFilePathTableMiniBatchDatasource( inputs, miniBatchSize );
                    end
                elseif isClassificationRNN && iIsSequenceInMemoryInput(inputs)
                    if executionSettings.useParallel
                        error( message( 'nnet_cnn:internal:cnn:DataDispatcherFactory:ParallelNotSupportedForLSTM' ) );
                    end
                    dispatcher = iCreateSequenceClassificationDispatcher( inputs, response, ...
                        miniBatchSize, sequenceLength, endOfEpoch, paddingValue, precision );
                    datasource = [];
                elseif isClassificationRNN && istable(inputs)
                    if executionSettings.useParallel
                        error( message( 'nnet_cnn:internal:cnn:DataDispatcherFactory:ParallelNotSupportedForLSTM' ) );
                    end
                    dispatcher = iCreateTableSequenceClassificationDispatcher( inputs, miniBatchSize, ...
                        sequenceLength, endOfEpoch, paddingValue, precision, layers );
                    datasource = [];
                else
                    error( message( 'nnet_cnn:internal:cnn:DataDispatcherFactory:InvalidData' ) );
                end
                
                if ~isa(datasource,'nnet.internal.cnn.DistributableMiniBatchDatasource') && executionSettings.useParallel
                   error( message( 'nnet_cnn:internal:cnn:DataDispatcherFactory:NonDistributableMiniBatchDatasource' ) );
                end
                
                if ~isempty(datasource)
                    dispatcher = nnet.internal.cnn.MiniBatchDatasourceDispatcher( datasource, miniBatchSize, endOfEpoch, precision );
                end

            end
                       
            % Move dispatch to the background if requested
            if isa( dispatcher, 'nnet.internal.cnn.BackgroundCapableDispatcher' ) && ...
                    dispatcher.RunInBackground
                if executionSettings.useParallel
                    error( message( 'nnet_cnn:internal:cnn:DataDispatcherFactory:BackgroundWithParallelNotSupported' ) );
                else
                    dispatcher = nnet.internal.cnn.BackgroundDispatcher( dispatcher );
                end
            end
            
            % Distribute for Parallel
            if executionSettings.useParallel
                retainDataOrder = isequal(shuffleOption, 'never');
                dispatcher = nnet.internal.cnn.DistributedDispatcher( dispatcher, executionSettings.workerLoad, retainDataOrder );
            end
        end
    end
end

function ds = iCreate4dArrayMiniBatchDatasource( inputs, response, miniBatchSize )
ds = nnet.internal.cnn.FourDArrayMiniBatchDatasource(inputs, response, miniBatchSize);
end

function ds = iCreateImageDatastoreMiniBatchDatasource( inputs, miniBatchSize )
ds = nnet.internal.cnn.ImageDatastoreMiniBatchDatasource( inputs, miniBatchSize );
end

function ds = iCreateFilePathTableMiniBatchDatasource( inputs, miniBatchSize )
ds = nnet.internal.cnn.FilePathTableMiniBatchDatasource( inputs, miniBatchSize );
end

function ds = iCreateInMemoryTableMiniBatchDatasource( inputs, miniBatchSize )
ds = nnet.internal.cnn.InMemoryTableMiniBatchDatasource( inputs, miniBatchSize );
end

function ds = iCreateSequenceClassificationDispatcher( inputs, response, ...
    miniBatchSize, sequenceLength, endOfEpoch, paddingValue, precision )
ds = nnet.internal.cnn.SequenceClassificationDispatcher( inputs, response, ...
    miniBatchSize, sequenceLength, endOfEpoch, paddingValue, precision );              
end

function ds = iCreateTableSequenceClassificationDispatcher( inputs, miniBatchSize, ...
    sequenceLength, endOfEpoch, paddingValue, precision, layers )
inputSize = layers(1).InputSize;
outputSize = layers(end).OutputSize;
ds = nnet.internal.cnn.TableSequenceClassificationDispatcher( inputs, miniBatchSize, ...
    sequenceLength, endOfEpoch, paddingValue, precision, inputSize, outputSize );            
end

function tf = iIsRealNumeric4DHostArray( x )
tf = iIsRealNumericData( x ) && iIsValidImageArray( x ) && ~iIsGPUArray( x );
end

function tf = iIsRealNumericData(x)
tf = isreal(x) && isnumeric(x) && ~issparse(x);
end

function tf = iIsValidImageArray(x)
% iIsValidImageArray   Return true if x is an array of
% one or multiple (colour or grayscale) images
tf = ( iIsGrayscale( x ) || iIsColour( x ) ) && ...
    iIs4DArray( x );
end

function tf = iIsGrayscale(x)
tf = size(x,3)==1;
end

function tf = iIsColour(x)
tf = size(x,3)==3;
end

function tf = iIs4DArray(x)
sz = size( x );
tf = numel( sz ) <= 4;
end

function tf = iIsGPUArray( x )
tf = isa(x, 'gpuArray');
end

function tf = iIsAnInMemoryTable( x )
firstCell = x{1,1};
tf = isnumeric( firstCell{:} );
end

function tf = iIsSequenceInMemoryInput( x )
tf = (iscell(x) && isvector(x) && all(cellfun(@isnumeric, x))) || ...
    (isnumeric(x) && ismatrix(x));
end

function tf = iIsSequenceClassificationLayers( x )
tf = ~isempty( x ) && isa( x(1), 'nnet.cnn.layer.SequenceInputLayer' ) && ...
    ( isa( x(end), 'nnet.cnn.layer.ClassificationOutputLayer' ) || ...
    isa( x(end), 'nnet.layer.ClassificationLayer' ) );
end