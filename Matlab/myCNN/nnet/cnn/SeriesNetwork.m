classdef SeriesNetwork
    % SeriesNetwork   A neural network with layers arranged in a series
    %
    %   A series network is one where the layers are arranged one after the
    %   other. There is a single input and a single output.
    %
    %   SeriesNetwork properties:
    %       Layers          - The layers of the network.
    %
    %   SeriesNetwork methods:
    %       predict         - Run the network on input data.
    %       classify        - Classify data with a network.
    %       activations     - Compute specific network layer activations.
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
    %   See also trainNetwork.
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties(Dependent, SetAccess = private)
        % Layers   The layers of the network
        %   The array of layers for the network. Each layer has different
        %   properties depending on what type of layer it is. The first
        %   layer is always an input layer, and the last layer is always an
        %   output layer.
        Layers
    end
    
    properties(Access = private)
        PrivateNetwork
    end
    
    properties(Access = private, Dependent)
        % InputSize    Size of the network input as stored in the input
        % layer
        InputSize
        
        % Outputsize    Size of the network output as stored in the output
        % layer
        OutputSize
    end
    
    methods
        function val = get.InputSize(this)
            val = this.PrivateNetwork.Layers{1}.InputSize;
        end
        
        function val = get.OutputSize(this)
            val = this.PrivateNetwork.Layers{end}.NumClasses;
        end
        
        function layers = get.Layers(this)
            layers = nnet.cnn.layer.Layer.createLayers(this.PrivateNetwork.Layers);
        end
    end
    
    methods(Access = public, Hidden)
        function this = SeriesNetwork(layers)
            % SeriesNetwork    Constructor for series network
            %
            % layers is an heterogeneous array of nnet.cnn.layer.Layer
            
            % Retrieve the internal layers
            internalLayers = nnet.cnn.layer.Layer.getInternalLayers(layers);
            
            % Create the network
            this.PrivateNetwork = nnet.internal.cnn.SeriesNetwork(internalLayers);
        end
    end
    
    methods(Access = public)
        function Y = predict(this, data, varargin)
            % predict   Make predictions on data with a network
            %
            %   Y = predict(net, X) will compute predictions of the network
            %   net on the data X. The format of X will depend on the input
            %   layer for the network.
            %
            %   For an image input layer, X may be:
            %       - A single image.
            %       - A 4D array of images, where the first three
            %         dimensions index the height, width and channels of an
            %         image, and the fourth dimension indexes the
            %         individual images.
            %       - An image datastore.
            %
            %   Y will contain the predicted scores, arranged in an N-by-K
            %   matrix, where N is the number of observations, and K is the
            %   number of classes.
            %
            %   Y = predict(net, X, 'PARAM1', VAL1, ...) will compute
            %   predictions with the following optional name/value pairs:
            %
            %       'MiniBatchSize'     - The size of the mini-batches for
            %                             computing predictions. Larger
            %                             mini-batch sizes lead to faster
            %                             predictions, at the cost of more
            %                             memory. The default is 128.
            %       'ExecutionEnvironment'
            %                           - The execution environment for the
            %                             network. This determines what 
            %                             hardware resources will be used 
            %                             to run the network.
            %                               - 'auto' - Use a GPU if it is
            %                                 available, otherwise use the 
            %                                 CPU.
            %                               - 'gpu' - Use the GPU. To use a
            %                                 GPU, you must have Parallel
            %                                 Computing Toolbox(TM), and a 
            %                                 CUDA-enabled NVIDIA GPU with 
            %                                 compute capability 3.0 or 
            %                                 higher. If a suitable GPU is 
            %                                 not available, predict 
            %                                 returns an error message.
            %                               - 'cpu' - Use the CPU.
            %                             The default is 'auto'.
            %
            %   See also SeriesNetwork/classify, SeriesNetwork/activations.
            
            % Set desired precision
            precision = nnet.internal.cnn.util.Precision('single');
            
            [miniBatchSize, executionEnvironment] = iParseAndValidatePredictInputs( varargin{:} );
            
            dispatcher = iDataDispatcher(data, miniBatchSize, precision);
            
            iAssertInputDataIsValidForPredict(dispatcher, this.InputSize);
            
            % Prepare the network for the correct prediction mode
            GPUShouldBeUsed = iGPUShouldBeUsed(executionEnvironment);
            if(GPUShouldBeUsed)
                this.PrivateNetwork = this.PrivateNetwork.setupNetworkForGPUPrediction();
            else
                this.PrivateNetwork = this.PrivateNetwork.setupNetworkForHostPrediction();
            end
            
            % Replace dropout layers with null layers
            predictNetwork = iReplaceDropouts(this.PrivateNetwork);
            
            % Use the dispatcher to run the network on the data
            Y = precision.cast( zeros([1 1 this.OutputSize dispatcher.NumObservations]) );
            dispatcher.start();
            while ~dispatcher.IsDone
                [X, ~, i] = dispatcher.next();
                
                if(GPUShouldBeUsed)
                    X = gpuArray(X);
                end
                
                Y(:,:,:,i) = gather(predictNetwork.predict(X));
            end
            
            Y = iFormatPredictionsAs2DRowResponses(Y);
        end
        
        function [labels, scores] = classify(this, X, varargin)
            % classify   Classify data with a network
            %
            %   [labels, scores] = classify(net, X) will classify the data
            %   X using the network net. labels will be an N-by-1
            %   categorical vector where N is the number of observations,
            %   and scores will be an N-by-K matrix where K is the number
            %   of output classes. The format of X will depend on the input
            %   layer for the network.
            %
            %   For an image input layer, X may be:
            %       - A single image.
            %       - A four dimensional numeric array of images, where the
            %         first three dimensions index the height, width, and
            %         channels of an image, and the fourth dimension
            %         indexes the individual images.
            %       - An image datastore.
            %
            %   [labels, scores] = classify(net, X, 'PARAM1', VAL1, ...)
            %   specifies optional name-value pairs described below:
            %
            %       'MiniBatchSize'     - The size of the mini-batches for
            %                             computing predictions. Larger
            %                             mini-batch sizes lead to faster
            %                             predictions, at the cost of more
            %                             memory. The default is 128.
            %       'ExecutionEnvironment'
            %                           - The execution environment for the
            %                             network. This determines what 
            %                             hardware resources will be used 
            %                             to run the network.
            %                               - 'auto' - Use a GPU if it is
            %                                 available, otherwise use the 
            %                                 CPU.
            %                               - 'gpu' - Use the GPU. To use a
            %                                 GPU, you must have Parallel
            %                                 Computing Toolbox(TM), and a 
            %                                 CUDA-enabled NVIDIA GPU with 
            %                                 compute capability 3.0 or 
            %                                 higher. If a suitable GPU is 
            %                                 not available, classify 
            %                                 returns an error message.
            %                               - 'cpu' - Use the CPU.
            %                             The default is 'auto'.
            %
            %   See also SeriesNetwork/predict, SeriesNetwork/activations.
            
            scores = this.predict( X, varargin{:} );
            labels = iUndummify( scores, iClassNames( this.PrivateNetwork ) );
        end
        
        function Y = activations(this, X, layerID, varargin)
            % activations   Computes network layer activations
            %
            %   Y = activations(net, X, layer) returns network
            %   activations for a specific layer. Network activations are
            %   computed by forward propagating input X through the network
            %   up to the specified layer. layer must be a numeric index
            %   or a character vector corresponding to one of the network
            %   layer names.
            %
            %   For a network with an image input layer, X may be:
            %       - A single image.
            %       - A 4D array of images, where the first three
            %         dimensions index the height, width and channels of an
            %         image, and the fourth dimension indexes the
            %         individual images.
            %       - An image datastore.
            %
            %   By default, Y will be a matrix with one row per
            %   observation. See the 'OutputAs' optional name-value pair
            %   for more details.
            %
            %   Y = activations(net, X, layer, 'PARAM1', VAL1, ...)
            %   specifies optional name-value pairs described below:
            %
            %       'OutputAs'    - A string specifying how to arrange the
            %                       output activations. Possible values
            %                       are given below:
            %                         - 'rows' - The output will be an
            %                           N-by-M matrix, where N is the
            %                           number of observations, and M is
            %                           the number of output elements from
            %                           the chosen layer. Each row of the
            %                           matrix is the output for a single
            %                           observation.
            %                         - 'columns' - The output will be an
            %                           M-by-N martrix, where M is the
            %                           number of output elements from the
            %                           chosen layer, and N is the number
            %                           of observations. Each column of the
            %                           matrix is the output for a single
            %                           observation.
            %                         - 'channels' - The output will be an
            %                           H-by-W-by-C-by-N array, where H, W
            %                           and C are the height, width and
            %                           number of channels for the output
            %                           of the chosen layer. Each
            %                           H-by-W-by-C sub-array is the output
            %                           for a single observation.
            %                       The default is 'rows'.
            %
            %       'MiniBatchSize' 
            %                     - The size of the mini-batches for
            %                       computing predictions. Larger
            %                       mini-batch sizes lead to faster
            %                       predictions, at the cost of more
            %                       memory. The default is 128.
            %
            %       'ExecutionEnvironment'
            %                     - The execution environment for the
            %                       network. This determines what hardware
            %                       resources will be used to run the
            %                       network.
            %                         - 'auto' - Use a GPU if it is
            %                           available, otherwise use the CPU.
            %                         - 'gpu' - Use the GPU. To use a
            %                           GPU, you must have Parallel
            %                           Computing Toolbox(TM), and a 
            %                           CUDA-enabled NVIDIA GPU with 
            %                           compute capability 3.0 or higher.
            %                           If a suitable GPU is not available,
            %                           activations returns an error 
            %                           message.
            %                         - 'cpu' - Use the CPU.
            %                       The default is 'auto'.
            %
            %   See also SeriesNetwork/predict, SeriesNetwork/classify.
            
            % Set desired precision
            precision = nnet.internal.cnn.util.Precision('single');
            
            internalLayers = this.PrivateNetwork.Layers;
            
            layerID = iValidateAndParseLayerID( layerID, internalLayers );
            
            [miniBatchSize, outputAs, executionEnvironment] = iParseAndValidateActivationsInputs( varargin{:} );
            
            dispatcher = iDataDispatcher( X, miniBatchSize, precision );
            
            iAssertInputDataIsValidForActivations( dispatcher, this.InputSize, outputAs );
            
            % Prepare the network for the correct prediction mode
            GPUShouldBeUsed = iGPUShouldBeUsed(executionEnvironment);
            if(GPUShouldBeUsed)
                this.PrivateNetwork = this.PrivateNetwork.setupNetworkForGPUPrediction();
            else
                this.PrivateNetwork = this.PrivateNetwork.setupNetworkForHostPrediction();
            end
            
            % Replace dropout layers with null layers
            predictNetwork = iReplaceDropouts( this.PrivateNetwork );
            
            inputSize = iImageSize( dispatcher );
            outputSize = iDetermineLayerOutputSize( internalLayers, layerID, inputSize );
            
            [sz, indexFcn, reshapeFcn] = iGetOutputSizeAndIndices(...
                outputAs, dispatcher.NumObservations, outputSize);
            
            % pre-allocate output buffer
            Y = precision.cast( zeros(sz) );
            
            % Use the dispatcher to run the network on the data
            dispatcher.start();
            while ~dispatcher.IsDone
                [X, ~, i] = dispatcher.next();
                
                if(GPUShouldBeUsed)
                    X = gpuArray(X);
                end
                
                indices = indexFcn(i);
                
                YChannelFormat = gather(predictNetwork.activations(X, layerID));
                
                Y(indices{:}) = reshapeFcn(YChannelFormat, numel(i));
            end
            
        end
        
        function out = saveobj(this)
            out.Version = 1.0;
            out.Layers = this.Layers; % User visible layers
        end
    end
    
    methods(Static)
        function this = loadobj(in)
            this = SeriesNetwork( in.Layers );
        end
    end
end

function layerIdx = iValidateAndParseLayerID(layerIdx, layers)
if ischar(layerIdx)
    name = layerIdx;
    
    [layerIdx, layerNames] = nnet.internal.cnn.layer.Layer.findLayerByName(layers, name);
    
    try
        % pretty print error message. will print available layer names in
        % case of a mismatch.
        validatestring(name, layerNames, 'activations','layer');
    catch Ex
        throwAsCaller(Ex);
    end
    
    % Only 1 match allowed. This is guaranteed during construction of SeriesNetwork.
    assert(numel(layerIdx) == 1);
else
    validateattributes(layerIdx, {'numeric'},...
        {'positive', 'integer', 'real', 'scalar', '<=', numel(layers)}, ...
        'activations', 'layer');
end
end

function outputSize = iDetermineLayerOutputSize(layers, layerIdx, inputSize)
% Determine output size of output layer.
if nargin<3
    inputSize = layers{1}.InputSize;
end
for i = 2:layerIdx
    inputSize = layers{i}.forwardPropagateSize(inputSize);
end
outputSize = inputSize;
end

function network = iReplaceDropouts(network)
% iReplaceDropouts    Replace all dropouts layer of a network with null
% layers with the same name
for i = 1:numel(network.Layers)
    if isa( network.Layers{i}, 'nnet.internal.cnn.layer.Dropout' )
        network.Layers{i} = nnet.internal.cnn.layer.NullLayer( network.Layers{i}.Name );
    end
end
end

function classNames = iClassNames( net )
classNames = net.Layers{end}.ClassNames;
end

function labels = iUndummify( scores, classNames )
labels = nnet.internal.cnn.util.undummify( scores, classNames );
end

function iAssertInputDataIsValidForPredict(dispatcher, inputSize)
if iDispatchesImagesOfSize(dispatcher, inputSize)
else
    exception = iCreateExceptionFromErrorID('nnet_cnn:SeriesNetwork:InvalidPredictDataForImageInputLayer');
    throwAsCaller(exception);
end
end

function iAssertInputDataIsValidForActivations( dispatcher, inputSize, outputAs )
% iAssertInputDataIsValidForActivations   Throws an error if the dispatcher doesn't given input that
% will work with the 'activations' method.

if iDispatchesImagesSmallerThan(dispatcher, inputSize)
    throwAsCaller( iCreateExceptionFromErrorID( 'nnet_cnn:SeriesNetwork:InvalidPredictDataForImageInputLayer' ) );
    
elseif iDispatchesImagesOfSize(dispatcher, inputSize)
    % The 'activations' method can use this data
    
elseif iLargerImagesCanBeOutputAs( outputAs )
    % The 'activations' method can use this data
else
    throwAsCaller( iCreateExceptionFromErrorID( 'nnet_cnn:SeriesNetwork:ActivationsLargeImagesOutputAs' ) );
end
end

function tf = iLargerImagesCanBeOutputAs( outputAs )
tf = isequal(outputAs, 'channels');
end

function iAssertMiniBatchSizeIsValid(x)
if(iIsPositiveIntegerScalar(x))
else
    exception = iCreateExceptionFromErrorID('nnet_cnn:SeriesNetwork:InvalidMiniBatchSize');
    throwAsCaller(exception);
end
end

function sz = iImageSize( x )
% iImageSize   Return the size of x as [H W C] where C is 1 when x is a
% grayscale image.
if iIsDataDispatcher( x )
    sz = x.ImageSize;
else
    sz = [size(x,1) size(x,2) size(x,3)];
end
end

function tf = iDispatchesImagesOfSize(dispatcher, inputSize)
dispatchedImageSize = iImageSize( dispatcher );
tf = isequal( dispatchedImageSize, inputSize );
end

function tf = iDispatchesImagesSmallerThan(dispatcher, inputSize)
% iDispatchesImagesSmallerThan   Check if the dispatcher dispatches images
% smaller than inputSize on either one of the first two dimensions or with
% different number of channels.
dispatchedImageSize = iImageSize( dispatcher );
tf = dispatchedImageSize(1) < inputSize(1) || ...
    dispatchedImageSize(2) < inputSize(2) || ...
    dispatchedImageSize(3) ~= inputSize(3);
end

function tf = iIsDataDispatcher(x)
tf = isa(x,'nnet.internal.cnn.DataDispatcher');
end

function tf = iIsPositiveIntegerScalar(x)
tf = all(x > 0) && iIsInteger(x) && isscalar(x);
end

function tf = iIsInteger(x)
tf = isreal(x) && isnumeric(x) && all(mod(x,1)==0);
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(errorID, getString(message(errorID, varargin{:})));
end

function dispatcher = iDataDispatcher(data, miniBatchSize, precision)
% iDataDispatcher   Use the factory to create a dispatcher.
try
    dispatcher = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcher( ...
        data, [], miniBatchSize, 'truncateLast', precision);
catch e
    iThrowInvalidDataException( e )
end
end

function iThrowInvalidDataException(e)
% iThrowInvalidDataException   Throws an InvalidData exception generated
% from DataDispatcherFactory as an InvalidPredictDataForImageInputLayer
% exception.
if (strcmp(e.identifier,'nnet_cnn:internal:cnn:DataDispatcherFactory:InvalidData'))
    exception = iCreateExceptionFromErrorID('nnet_cnn:SeriesNetwork:InvalidPredictDataForImageInputLayer');
    throwAsCaller(exception)
else
    rethrow(e)
end
end

function [miniBatchSize, executionEnvironment] = iParseAndValidatePredictInputs(varargin)
p = inputParser;

defaultMiniBatchSize = iGetDefaultMiniBatchSize();
defaultExecutionEnvironment = 'auto';

addParameter(p, 'MiniBatchSize', defaultMiniBatchSize);
addParameter(p, 'ExecutionEnvironment', defaultExecutionEnvironment);

parse(p, varargin{:});
iAssertMiniBatchSizeIsValid(p.Results.MiniBatchSize);

miniBatchSize = p.Results.MiniBatchSize;
executionEnvironment = iValidateExecutionEnvironment(p.Results.ExecutionEnvironment, 'predict');
end

function [miniBatchSize, outputAs, executionEnvironment] = iParseAndValidateActivationsInputs(varargin)
p = inputParser;

defaultMiniBatchSize = iGetDefaultMiniBatchSize();
defaultOutputAs = 'rows';
defaultExecutionEnvironment = 'auto';

addParameter(p, 'MiniBatchSize', defaultMiniBatchSize);
addParameter(p, 'OutputAs', defaultOutputAs);
addParameter(p, 'ExecutionEnvironment', defaultExecutionEnvironment);

parse(p, varargin{:});

iAssertMiniBatchSizeIsValid(p.Results.MiniBatchSize);

miniBatchSize = p.Results.MiniBatchSize;
outputAs = iValidateOutputAs(p.Results.OutputAs);
executionEnvironment = iValidateExecutionEnvironment(p.Results.ExecutionEnvironment, 'activations');
end

function [outputBatchSize, indexFcn, reshapeFcn] = iGetOutputSizeAndIndices(outputAs, numObs, outputSize)
% Returns the output batch size, indexing function, and reshaping function.
% The indexing function provides the right set of indices based on the
% 'OutputAs' setting. The reshaping function reshapes channel
% formatted output to the shape required for the 'OutputAs' setting.
switch outputAs
    case 'rows'
        outputBatchSize = [numObs prod(outputSize)];
        indexFcn = @(i){i 1:prod(outputSize)};
        reshapeFcn = @(y,n)transpose(reshape(y, [], n));
    case 'columns'
        outputBatchSize = [prod(outputSize) numObs];
        indexFcn = @(i){1:prod(outputSize) i};
        reshapeFcn = @(y,n)reshape(y, [], n);
    case 'channels'
        outputBatchSize = [outputSize numObs];
        indices = arrayfun(@(x)1:x, outputSize, 'UniformOutput', false);
        indexFcn = @(i)[indices i];
        reshapeFcn = @(y,~)y;
end
end

function val = iGetDefaultMiniBatchSize
val = 128;
end

function valid = iValidateOutputAs(str)
validChoices = {'rows', 'columns', 'channels'};
valid = validatestring(str, validChoices, 'activations', 'OutputAs');
end

function validString = iValidateExecutionEnvironment(inputString, caller)
validExecutionEnvironments = {'auto', 'gpu', 'cpu'};
validString = validatestring(inputString, validExecutionEnvironments, caller, 'ExecutionEnvironment');
end

function YFormatted = iFormatPredictionsAs2DRowResponses(Y)
YFormatted = shiftdim(Y);
YFormatted = YFormatted';
end

function tf = iGPUShouldBeUsed(executionEnvironment)
switch executionEnvironment
    case 'cpu'
        tf = false;
    case 'gpu'
        if(nnet.internal.cnngpu.isGPUCompatible(true))
            tf = true;
        else
            error(message('nnet_cnn:internal:cnngpu:GPUArchMismatch'));
        end
    case 'auto'
        if(nnet.internal.cnngpu.isGPUCompatible(false))
            tf = true;
        else
            tf = false;
        end
    otherwise
        error(message('nnet_cnn:SeriesNetwork:InvalidExecutionEnvironment'));
end
end