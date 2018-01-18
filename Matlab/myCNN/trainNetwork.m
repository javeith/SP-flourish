function [trainedNet, info] = trainNetwork(varargin)
% trainNetwork   Train a neural network
%
%   trainedNet = trainNetwork(ds, layers, options) trains and returns a
%   network trainedNet. ds is an imageDatastore with categorical labels,
%   layers is an array of network layers, and options is a set of training
%   options.
%
%   trainedNet = trainNetwork(X, Y, layers, options) trains and returns a
%   network trainedNet. The format for X will depend on the input layer.
%   For an image input layer, X is a numeric array of images arranged so
%   that the first three dimensions are the height, width and channels, and
%   the last dimension indexes the individual images. Y is a vector of
%   categorical responses. 
%
%   [trainedNet, info] = trainNetwork(...) trains and returns a network
%   trainedNet. info contains information on training progress.
%
%   Example:
%       Train a convolutional neural network on some synthetic images
%       of handwritten digits. Then run the trained network on a test
%       set, and calculate the accuracy.
%
%       [XTrain, TTrain] = digitTrain4DArrayData;
%
%       layers = [ ...
%           imageInputLayer([28 28 1])
%           convolution2dLayer(5,20)
%           reluLayer()
%           maxPooling2dLayer(2,'Stride',2)
%           fullyConnectedLayer(10)
%           softmaxLayer()
%           classificationLayer()];
%       options = trainingOptions('sgdm');
%       net = trainNetwork(XTrain, TTrain, layers, options);
%
%       [XTest, TTest] = digitTest4DArrayData;
%
%       YTest = classify(net, XTest);
%       accuracy = sum(YTest == TTest)/numel(TTest)
%
%   See also nnet.cnn.layer, trainingOptions.

%   Copyright 2015-2016 The MathWorks, Inc.

nnet.internal.cnngpu.isGPUCompatible(true);

narginchk(3,4);
[layers, opts, X, Y] = iParseInput(varargin{:});

% Set desired precision
precision = nnet.internal.cnn.util.Precision('single');

% Create a dispatcher
dispatcher = iCreateTrainingDataDispatcher(X, Y, opts, precision);

% Retrieve the classnames from the dispatcher.
classNames = dispatcher.ClassNames;

% Do inference on layers
layers = nnet.cnn.layer.Layer.inferParameters(layers);

% Check the output size matches the number of classes
iAssertLastLayerPredictsCorrectNumClasses(layers,classNames);

% Retrieve the internal layers
internalLayers = nnet.cnn.layer.Layer.getInternalLayers(layers);

% Check that the input data has a valid size
iAssertThatTrainingDataHasValidSizeForTraining(internalLayers{1}, dispatcher);

% Initialize learnable parameters
internalLayers = iInitializeParameters(internalLayers, precision);

% Store labels into cross entropy layer
internalLayers = iStoreClassNames(internalLayers, classNames);

% Create the network
trainedNet = nnet.internal.cnn.SeriesNetwork(internalLayers);

% Convert learnable parameters to the correct format
trainedNet = trainedNet.prepareNetworkForTraining();

% Do pre-processing work required for normalizing data
trainedNet = iInitializeNetworkNormalizations(trainedNet, X, Y, opts, precision);

% Instantiate reporters as needed
reporters = iOptionalReporters(opts);
% Always create the info recorder (becuase we will reference it later) but
% only add it to the list of reporters if actually needed.
infoRecorder = nnet.internal.cnn.util.TrainingInfoRecorder();
if nargout >= 2
    reporters.add( infoRecorder  );
end

% Create a trainer to train the network with dispatcher and options
trainer = nnet.internal.cnn.Trainer(opts, precision, reporters);

try
    trainedNet = trainer.train(trainedNet, dispatcher);
catch e
    iRethrowInternalException( e );
end

% Return arguments
trainedNet = iInternalNetworkToExternal(trainedNet);
info = infoRecorder.Info;
end

function iRethrowInternalException( e )
externalErrorID = strrep(e.identifier, ':internal', '');
exception = MException(externalErrorID, e.message);
throwAsCaller(exception)
end

function externalNetwork = iInternalNetworkToExternal(internalNetwork)
internalNetwork = internalNetwork.prepareNetworkForPrediction();
externalNetwork = SeriesNetwork(iExternalLayers(internalNetwork.Layers));
end

function layers = iInitializeParameters(layers, precision)
for i = 1:numel(layers)
    layers{i} = layers{i}.initializeLearnableParameters(precision);
end
end

function iAssertThatTrainingDataHasValidSizeForTraining(internalInputLayer, dispatcher)
if(~internalInputLayer.isValidTrainingImageSize(dispatcher.ImageSize))
    error(message('nnet_cnn:trainNetwork:TrainingImagesInvalidSize'));
end
end

function iAssertLastLayerPredictsCorrectNumClasses(layers,classNames)
outputSize = layers(end).OutputSize;
numClasses = numel( classNames );

if ~isequal( outputSize, numClasses )
    exception = iException( 'nnet_cnn:trainNetwork:OutputSizeNumClassesMismatch', ...
        mat2str( outputSize ), mat2str( numClasses ) );
    throwAsCaller( exception );
end
end

function exception = iException( id, varargin )
msg = getString( message( id, varargin{:} ) );
exception = MException( id, msg );
end

function externalLayers = iExternalLayers(internalLayers)
externalLayers = nnet.cnn.layer.Layer.createLayers(internalLayers);
end

function layers = iStoreClassNames(layers, labels)
layers{end}.ClassNames = labels;
end


function iAssertInputIsValid(X)
% input may be an imageDatastore or a custom dispatcher. The custom
% dispatcher api is for internal use only.
if ~isa(X, 'nnet.internal.cnn.DataDispatcher')     
    iAssertValidImageDatastore( X );
    iAssertDatastoreHasLabels( X );
    iAssertDatastoreLabelsAreCategorical( X );
    iAssertLabelsAreDefined( X.Labels );
end
end

function [layers, opts, X, Y] = iParseInput(varargin)
switch nargin
    case 3
        X = varargin{1};
        Y = [];
        layers = varargin{2};
        opts = varargin{3};
        iAssertInputIsValid( X );        
    case 4
        X = varargin{1};
        Y = varargin{2};
        layers = varargin{3};
        opts = varargin{4};
        iAssertValidImageArray( X );
        iAssertCategoricalResponseVector( Y );
        iAssertLabelsAreDefined( Y );
        iAssertXAndYHaveSameNumberOfObservations( X, Y );      
    otherwise
        error(message('nnet_cnn:trainNetwork:TooManyInputArguments'));
end

% Verify layers and options
iAssertValidLayerArray(layers);
iAssertValidTrainingOptions(opts)
end

function dispatcher = iCreateTrainingDataDispatcher(X, Y, opts, precision)
% Create a dispatcher.
dispatcher = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcher( ...
    X, Y, opts.MiniBatchSize, 'discardLast', precision);
end

function net = iInitializeNetworkNormalizations(net, X, Y, opts, precision)

% Always use 'truncateLast' as we want to process only the data we have.
data = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcher( ...
    X, Y, opts.MiniBatchSize, 'truncateLast', precision);

augmentations = iGetAugmentations(net);
normalization = iGetNormalization(net);

zerocenter = arrayfun(@(x)isa(x,...
    'nnet.internal.cnn.layer.ZeroCenterImageTransform'), normalization);

if any(zerocenter)
    assert(sum(zerocenter == 1)==1, 'There should only be 1 zero center');
    avgI = iComputeAverageImage(data, augmentations);
    net.Layers{1}.AverageImage = precision.cast(avgI);
end
end

function a = iGetAugmentations(net)
a = net.Layers{1}.TrainTransforms;
end

function n = iGetNormalization(net)
n = net.Layers{1}.Transforms;
end

function avgI = iComputeAverageImage(data, augmentations)
% Average image is computed on the GPU
data.start();
accum = gpuArray(0);
numImages = 0;
while ~data.IsDone
    X = data.next();
    X = apply(augmentations, X);
    X = gpuArray(double(X));    
    accum = accum + sum(X, 4);
    numImages = numImages + size(X,4);
end

avgI = accum./numImages;
avgI = gather(avgI);
end

function reporter = iOptionalReporters(opts)
% iOptionalReporters   Create a vector of Reporters based on the given
% training options
%
% See also: nnet.internal.cnn.util.VectorReporter
reporter = nnet.internal.cnn.util.VectorReporter();

if opts.Verbose
    reporter.add( nnet.internal.cnn.util.ProgressDisplayer() );
end

if ~isempty( opts.CheckpointPath )
    checkpointSaver = nnet.internal.cnn.util.CheckpointSaver( opts.CheckpointPath );
    checkpointSaver.ConvertorFcn = @iInternalNetworkToExternal;
    reporter.add( checkpointSaver );
end
end

function iAssertValidImageDatastore(imds)
if ~iIsImageDatastore(imds)
    error(message('nnet_cnn:trainNetwork:NotAnImageDatastore'))
end
end

function iAssertDatastoreHasLabels(imds)
if isempty(imds.Labels)
    error(message('nnet_cnn:trainNetwork:ImageDatastoreHasNoLabels'))
end
end

function iAssertDatastoreLabelsAreCategorical(imds)
if ~iscategorical(imds.Labels)
    error(message('nnet_cnn:trainNetwork:ImageDatastoreMustHaveCategoricalLabels'))
end
end

function iAssertLabelsAreDefined(labels)
if(any(isundefined(labels)))
    error(message('nnet_cnn:trainNetwork:UndefinedLabels'));
end
end

function iAssertValidImageArray(x)
if isa(x, 'gpuArray') || ~isnumeric(x) || ~isreal(x) || ~iIsValidImageArray(x)
    error(message('nnet_cnn:trainNetwork:XIsNotValidImageArray'))
end
end

function iAssertCategoricalResponseVector(x)
if ~iscategorical(x)
    error(message('nnet_cnn:trainNetwork:YIsNotCategoricalResponseVector'))
end
end

function iAssertXAndYHaveSameNumberOfObservations(x, y)
if size(x,4)~=size(y,1)
    error(message('nnet_cnn:trainNetwork:XAndYHaveDifferentObservations'))
end
end

function tf = iIsImageDatastore(x)
tf = isa(x, 'matlab.io.datastore.ImageDatastore');
end

function tf = iIsValidImageArray(x)
% iIsValidImageArray   Return true if x is an array of
% one or multiple (colour or grayscale) images
tf = iIsRealNumericData( x ) && ...
    ( iIsGrayscale( x ) || iIsColour( x ) ) && ...
    iIs4DArray( x );
end

function tf = iIsRealNumericData(x)
tf = isreal(x) && isnumeric(x);
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

function iAssertValidLayerArray(x)
if ~isa(x, 'nnet.cnn.layer.Layer')
    error(message('nnet_cnn:trainNetwork:InvalidLayersArray'))
end
end

function iAssertValidTrainingOptions(x)
if ~isa(x, 'nnet.cnn.TrainingOptionsSGDM')
    error(message('nnet_cnn:trainNetwork:InvalidTrainingOptions'))
end
end
