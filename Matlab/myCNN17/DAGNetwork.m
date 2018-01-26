classdef DAGNetwork
    % DAGNetwork   A Directed Acyclic Graph Network
    %
    %   A DAG network can have its layers arranged in a directed acyclic
    %   graph.
    %
    %   DAGNetwork properties:
    %       Layers          - The layers of a network
    %       Connections     - The connections between the layers
    %
    %   DAGNetwork methods:
    %       predict         - Run the network on input data
    %       classify        - Classify data with a network
    %       plot            - Plot a diagram of the network
    %
    %   Example:
    %       Train a network on synthetic digit data, and measure its
    %       accuracy:
    %
    %       [XTrain, YTrain] = digitTrain4DArrayData;
    %
    %       layers = [
    %           imageInputLayer([28 28 1], 'Name', 'input')
    %           convolution2dLayer(5, 20, 'Name', 'conv_1')
    %           reluLayer('Name', 'relu_1')
    %           convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_2')
    %           reluLayer('Name', 'relu_2')
    %           convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_3')
    %           reluLayer('Name', 'relu_3')
    %           additionLayer(2, 'Name', 'add')
    %           fullyConnectedLayer(10, 'Name', 'fc')
    %           softmaxLayer('Name', 'softmax')
    %           classificationLayer('Name', 'classoutput')];
    %
    %       lgraph = layerGraph(layers);
    %
    %       lgraph = connectLayers(lgraph, 'relu_1', 'add/in2');
    %
    %       plot(lgraph);
    %
    %       options = trainingOptions('sgdm');
    %       [net,info] = trainNetwork(XTrain, YTrain, lgraph, options);
    %
    %       [XTest, YTest] = digitTest4DArrayData;
    %       YPred = classify(net, XTest);
    %       accuracy = sum(YTest == YPred)/numel(YTest)
    %
    %   See also LayerGraph.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Access = private)
        % PrivateNetwork
        PrivateNetwork
        
        % LayersMap
        LayersMap        
    end
    
    properties(Dependent, SetAccess = private)
        % Layers   The layers in the network
        %   An array of the layers in the network. Each layer has different
        %   properties depending on what type of layer it is.
        Layers
        
        % Connections   A table of connections between the layers
        %   A table with one row for each connection between two layers. 
        %   The table has two columns:
        %       Source          - The name of the layer (and the layer 
        %                         output if applicable) where the 
        %                         connection begins.
        %       Destination     - The name of the layer (and the layer 
        %                         input if applicable) where the connection
        %                         ends.
        Connections
    end
    
    properties(Dependent, Hidden, SetAccess = private)
        % SortedLayers   Layers in a topologically sorted order
        %   An array of the layers in the network. Each layer has different
        %   properties depending on what type of layer it is.
        SortedLayers
        
        % SortedConnections   Connections in a topologically sorted order
        %   A table with one row for each connection between two layers.
        %   The table has two columns:
        %       Source          - The name of the layer (and the layer
        %                         output if applicable) where the
        %                         connection begins.
        %       Destination     - The name of the layer (and the layer
        %                         input if applicable) where the connection
        %                         ends.
        SortedConnections
    end
    
    properties(Dependent, Access = private)
        % Internal LayerGraph with trained values of learnable parameters
        LayerGraph
    end
    
    methods
        function val = get.Layers(this)
            % Using OriginalLayers from PrivateNetwork.
            val = this.LayersMap.externalLayers( this.PrivateNetwork.OriginalLayers );
        end
        
        function val = get.Connections(this)
            % Using OriginalLayers and OriginalConnections from
            % PrivateNetwork.
            val = nnet.internal.cnn.util.internalToExternalConnections( ...
                this.PrivateNetwork.OriginalConnections, ...
                this.Layers);
        end
        
        function val = get.SortedLayers(this)
            % Using topologically sorted Layers from PrivateNetwork.
            val = this.LayersMap.externalLayers( this.PrivateNetwork.Layers );
        end
        
        function val = get.SortedConnections(this)
            % Using topologically sorted Layers and Connections from
            % PrivateNetwork.
            val = nnet.internal.cnn.util.internalToExternalConnections( ...
                this.PrivateNetwork.Connections, ...
                this.SortedLayers);
        end
        
        function val = get.LayerGraph(this)
            val = this.PrivateNetwork.LayerGraph;
        end
    end
    
    methods(Hidden)
        function this = DAGNetwork(internalDAGNetwork, layersMap)
            % DAGNetwork   Constructor for the DAG network

            this.PrivateNetwork = internalDAGNetwork;
            this.LayersMap = layersMap;                        
        end
    end
    
    methods(Hidden)
        function layerGraph = getLayerGraph(this)
            % getLayerGraph   Get an internal layer graph for this
            % network after training
            layerGraph = this.LayerGraph;
        end
        
        function layersMap = getLayerMap(this)
            % getLayersMap   Get the layer map for this network
            layersMap = this.LayersMap;
        end
        
        function Y = activations(this, X, layerNameOrIndex, varargin)
            % activations   Computes network layer activations
            %
            %   Y = activations(net, X, layer) returns network
            %   activations for a specific layer. Network activations are
            %   computed by forward propagating input X through the network
            %   up to the specified layer. layer must be a numeric index
            %   or a character vector corresponding to one of the network
            %   layer names in the 'Layers' property.
            %
            %   For a network with a single image input layer, X may be:
            %       - A 4D array of images, where the first three
            %         dimensions index the height, width and channels of an
            %         image, and the fourth dimension indexes the
            %         individual images.
            %       - An image datastore.
            %       - A table, where the first column contains either image
            %         paths or images.
            %
            %   For a network with N image input layers, X is a 1-by-N cell
            %   array, where each entry of the cell array maybe any of the
            %   things mentioned above. Note that:
            %       - The number of observations that is passed in for each
            %         input layer MUST match (e.g. if there are two inputs,
            %         you cannot pass in two images for one input, and
            %         three for the other).
            %       - The order of the data in the cell array should
            %         reflect the order of the input layers in the 'Layers'
            %         property.
            %
            %   Y will be an H-by-W-by-C-by-N array, where H, W and C are
            %   the height, width and number of channels for the first
            %   output of the chosen layer. Each H-by-W-by-C sub-array is
            %   the output for a single observation.
            %
            %   The activations function is not supported for networks with
            %   a sequence input layer.
            %
            %   Y = activations(net, X, layer, 'PARAM1', VAL1, ...)
            %   specifies optional name-value pairs described below:
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
            %   See also DAGNetwork/predict, DAGNetwork/classify.
            
            % NOTE: Input layerNameOrIndex is specified with respect to the
            % Layers property which holds the external version of the
            % topologically sorted internal layers.
            
            % Set the desired precision
            precision = nnet.internal.cnn.util.Precision('single');
            
            % Validate optional inputs
            [miniBatchSize, executionEnvironment] = iParseAndValidateActivationsNameValuePairs(varargin{:});
            
            % Validate layerIndex
            internalLayers = this.PrivateNetwork.Layers;
            layerIndex = nnet.internal.cnn.util.validateNetworkLayerNameOrIndex(layerNameOrIndex, internalLayers, 'activations');
            
            % Validate X
            iValidateInputDataMatchesNumberOfInputLayers(X, this.PrivateNetwork.NumInputLayers);
            
            % Create dispatchers
            dispatchers = iCreateDataDispatchersForEachInputLayer(X, miniBatchSize, precision);
            
            % Validate that input sizes from dispatchers are suitable for
            % computing activations
            iValidateDispatchersForActivations(dispatchers, this.PrivateNetwork.InputSizes);
            
            % Ensure that all dispatchers have the same number of
            % observations
            iValidateDispatchersHaveSameNumberOfObservations(dispatchers);
            
            % Common number of observations in each dispatcher
            numObservations = dispatchers{1}.NumObservations;
            
            % Prepare the network for the correct prediction mode
            GPUShouldBeUsed = nnet.internal.cnn.util.GPUShouldBeUsed(executionEnvironment);
            if(GPUShouldBeUsed)
                this.PrivateNetwork = this.PrivateNetwork.setupNetworkForGPUPrediction();
            else
                this.PrivateNetwork = this.PrivateNetwork.setupNetworkForHostPrediction();
            end
            
            % Get input sizes from dispatchers
            dispatcherInputSizes = iGetDispatcherInputSizes(dispatchers);
            
            % Figure out activation sizes for specified layerIndex.
            % activationSizes below is a cell array such that
            % activationSizes{i} is the size for activation i from the
            % layer corresponding to layerIndex
            layerOutputSizes = inferOutputSizesGivenInputSizes(this.PrivateNetwork, dispatcherInputSizes);
            activationSizes = layerOutputSizes{layerIndex};
            activationSizes = iWrapInCell(activationSizes);
            
            % Allocate space for activations from the specified layer. The
            % output sizes calculated by MaxPooling2D layer are not
            % consistent with outputs from it's predict method. So, for
            % now, we only return the first activation for the specified
            % layerIndex. In the future, to get all activations, we can set
            % numActivations = numel(activationSizes) on the next line
            numActivations = 1;
            Y = cell(1,numActivations);
            for i = 1:numActivations
                Y{i} = precision.cast( zeros([activationSizes{i} numObservations]) );
            end
            
            % Start dispatchers
            dispatchers = iStartDispatchers(dispatchers);
            
            % Get activations one by one for each minibatch
            while ( ~dispatchers{1}.IsDone )
                [X, indices] = iNextMiniBatch(dispatchers);
                
                if (GPUShouldBeUsed)
                    for i = 1:numel(X)
                        X{i} = gpuArray(X{i});
                    end
                end
                
                YBatch = this.PrivateNetwork.activations(X, layerIndex);
                for i = 1:numActivations
                    Y{i}(:,:,:,indices) = gather(YBatch{i});
                end
            end
            
            % Don't return Y as a length 1 cell array in case we have only
            % one activation
            if ( numActivations == 1 )
                Y = Y{1};
            end
        end
        
        function topologicalOrder = getTopologicalOrder(this)
            topologicalOrder = this.PrivateNetwork.TopologicalOrder;
        end
    end
    
    methods
        function Y = predict(this, X, varargin)
            % predict   Make predictions on data with network
            %
            %   Y = predict(net, X) will compute predictions of the network
            %   net on the data X. The format of X will depend on the input
            %   layer for the network.
            %
            %   For a network with a single image input layer, X may be:
            %       - A single image.
            %       - A 4D array of images, where the first three
            %         dimensions index the height, width and channels of an
            %         image, and the fourth dimension indexes the
            %         individual images.
            %       - An image datastore.
            %       - A table, where the first column contains either image 
            %         paths or images.
            %
            %   For a network with N image input layers, X is a 1-by-N cell 
            %   array, where each entry of the cell array maybe any of the
            %   things mentioned above. Note that:
            %       - The number of observations that is passed in for each
            %         input layer MUST match (e.g. if there are two inputs,
            %         you cannot pass in two images for one input, and
            %         three for the other).
            %       - The order of the data in the cell array should
            %         reflect the order of the input layers in the 'Layers'
            %         property.
            %
            %   For a classification problem, Y will contain the predicted
            %   scores, arranged in an N-by-K matrix, where N is the number
            %   of observations, and K is the number of classes.
            %
            %   For a regression problem, Y will contain the predicted
            %   responses, arranged in an N-by-R matrix, where N is the
            %   number of observations, and R is the number of responses,
            %   or in an H-by-W-by-C-by-N 4D array, where N is the number
            %   of observations and H-by-W-by-C is the size of a single
            %   response.
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
            
            % Set desired precision
            precision = nnet.internal.cnn.util.Precision('single');
            
            [miniBatchSize, executionEnvironment] = iParseAndValidatePredictNameValuePairs( varargin{:} );
            
            iValidateInputDataMatchesNumberOfInputLayers(X, this.PrivateNetwork.NumInputLayers);
            
            dispatchers = iCreateDataDispatchersForEachInputLayer(X, miniBatchSize, precision);
            
            iValidateInputDataMatchesSizesOfInputLayers(dispatchers, this.PrivateNetwork.InputSizes);
            iValidateDispatchersHaveSameNumberOfObservations(dispatchers);
            
            % Prepare the network for the correct prediction mode
            GPUShouldBeUsed = nnet.internal.cnn.util.GPUShouldBeUsed(executionEnvironment);
            if(GPUShouldBeUsed)
                this.PrivateNetwork = this.PrivateNetwork.setupNetworkForGPUPrediction();
            else
                this.PrivateNetwork = this.PrivateNetwork.setupNetworkForHostPrediction();
            end
            
            % Validate that the dispatchers have the same number of
            % observations
            iValidateDispatchersHaveSameNumberOfObservations(dispatchers);
            numObservations = dispatchers{1}.NumObservations;
            
            % Allocate space for the output data.
            numOutputLayers = this.PrivateNetwork.NumOutputLayers;
            Y = cell(1, numOutputLayers);
            for i = 1:numOutputLayers
                Y{i} = precision.cast( zeros([this.PrivateNetwork.OutputSizes{i} numObservations]) );
            end
            
            % Use the dispatcher to run the network on the data
            dispatchers = iStartDispatchers(dispatchers);
            
            while ~dispatchers{1}.IsDone
                [X, indices] = iNextMiniBatch(dispatchers);
                
                if(GPUShouldBeUsed)
                    for i = 1:numel(X)
                        X{i} = gpuArray(X{i});
                    end
                end
                
                YBatch = this.PrivateNetwork.predict(X);
                for i = 1:numOutputLayers
                    Y{i}(:,:,:,indices) = gather(YBatch{i});
                end
            end
            
            for i = 1:numOutputLayers
                Y{i} = iFormatPredictionsAs2DRowResponses(Y{i});
            end
            
            if numOutputLayers == 1
                Y = Y{1};
            end
        end
        
        function [labelsToReturn, scoresToReturn] = classify(this, X, varargin)
            % classify   Classify data with the network
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
            %       - A table, where the first column contains either image 
            %         paths or images.
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
            %   See also DAGNetwork/predict, DAGNetwork/activations.
            
            scores = this.predict( X, varargin{:} );
            scores = {scores};
            
            labelsToReturn = {};
            scoresToReturn = {};
            for i = 1:this.PrivateNetwork.NumOutputLayers
                if(iIsClassificationOutputLayer(this.PrivateNetwork.Layers{this.PrivateNetwork.OutputLayerIndices(i)}))
                    
                    classNames = this.PrivateNetwork.Layers{this.PrivateNetwork.OutputLayerIndices(i)}.ClassNames;
                    categories = categorical(classNames);
                    categories = setcats(categories, classNames);
                    
                    labels = iUndummify(scores{i}, categories);
                    labelsToReturn = [labelsToReturn {labels}]; %#ok<AGROW>
                    scoresToReturn = [scoresToReturn scores{i}]; %#ok<AGROW>
                end
            end
            
            % If there are no labels returned, classify should error
            if isempty(labelsToReturn)
                exception = iCreateExceptionFromErrorID('nnet_cnn:DAGNetwork:InvalidNetworkForClassify');
                throwAsCaller(exception);
            end
            
            if this.PrivateNetwork.NumOutputLayers == 1
                labelsToReturn = labelsToReturn{1};
                scoresToReturn = scoresToReturn{1};
            end
        end
        
        function plot(this)
            % plot   Plot a diagram of the DAG network
            %
            %   plot(net) plots a diagram of the DAG network net. Each
            %   layer in the diagram is labelled by its name.
            
            this.LayerGraph.plot();
        end
        
        function out = saveobj(this)
            out.Version = 1.0;
            out.Layers = this.Layers;
            out.Connections = this.Connections;
        end
    end
    
    methods(Static)
        function this = loadobj(in)
            this = DAGNetwork.fromLayersAndConnections(in.Layers, in.Connections);
        end
    end
    
    methods(Static, Access = private)
        function this = fromLayersAndConnections(externalLayers, externalConnections)
            % Get internal layers
            internalLayers = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers(externalLayers);
            % Get internal connections
            internalConnections = nnet.internal.cnn.util.externalToInternalConnections(externalConnections, externalLayers);
            % Create an internal layer graph
            internalLayerGraph = nnet.internal.cnn.LayerGraph(internalLayers, internalConnections);
            % Sort the internal layer graph
            [internalLayerGraph, topologicalOrder] = toposort(internalLayerGraph);
            % Create an internal DAG network
            internalDAGNetwork = nnet.internal.cnn.DAGNetwork(internalLayerGraph, topologicalOrder);
            % Create an external DAG network
            layersMap = nnet.internal.cnn.layer.util.InternalExternalMap(externalLayers);
            this = DAGNetwork(internalDAGNetwork, layersMap);
        end
    end
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

function [miniBatchSize, executionEnvironment] = iParseAndValidatePredictNameValuePairs(varargin)
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

function [miniBatchSize, executionEnvironment] = iParseAndValidateActivationsNameValuePairs(varargin)
p = inputParser;

defaultMiniBatchSize = iGetDefaultMiniBatchSize();
defaultExecutionEnvironment = 'auto';

addParameter(p, 'MiniBatchSize', defaultMiniBatchSize);
addParameter(p, 'ExecutionEnvironment', defaultExecutionEnvironment);

parse(p, varargin{:});
iAssertMiniBatchSizeIsValid(p.Results.MiniBatchSize);

miniBatchSize = p.Results.MiniBatchSize;
executionEnvironment = iValidateExecutionEnvironment(p.Results.ExecutionEnvironment, 'activations');
end

function iAssertMiniBatchSizeIsValid(x)
if(iIsPositiveIntegerScalar(x))
else
    exception = iCreateExceptionFromErrorID('nnet_cnn:SeriesNetwork:InvalidMiniBatchSize');
    throwAsCaller(exception);
end
end

function val = iGetDefaultMiniBatchSize
val = 128;
end

function validString = iValidateExecutionEnvironment(inputString, caller)
validExecutionEnvironments = {'auto', 'gpu', 'cpu'};
validString = validatestring(inputString, validExecutionEnvironments, caller, 'ExecutionEnvironment');
end

function iValidateInputDataMatchesNumberOfInputLayers(X, numInputLayers)
if(iscell(X))
    iValidateIsOneByNCell(X, numInputLayers);
end
end

function iValidateIsOneByNCell(x, n)
validateattributes(x, {'cell'}, {'size',[1 n]});
end

function dispatchers = iCreateDataDispatchersForEachInputLayer(X, miniBatchSize, precision)
if(iscell(X))
    % Loop and create multiple dispatchers
    numDispatchers = numel(X);
    dispatchers = cell(1, numDispatchers);
    for i = 1:numDispatchers
        dispatchers{i} = iCreateDataDispatcher(X{i}, miniBatchSize, precision);
    end
else
    % Create a single dispatcher
    dispatchers = { iCreateDataDispatcher(X, miniBatchSize, precision) };
end
end

function dispatcher = iCreateDataDispatcher(X, miniBatchSize, precision)
try
    dispatcher = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcher( ...
        X, [], miniBatchSize, 'truncateLast', precision);
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

function iValidateInputDataMatchesSizesOfInputLayers(dispatchers, inputSizes)
numInputLayers = numel(inputSizes);
for i = 1:numInputLayers
    iValidateDispatcherHasThisInputSize(dispatchers{i}, inputSizes{i});
end
end

function iValidateDispatcherHasThisInputSize(dispatcher, inputSize)
if isequal(dispatcher.ImageSize, inputSize)
else
    exception = iCreateExceptionFromErrorID('nnet_cnn:SeriesNetwork:InvalidPredictDataForImageInputLayer');
    throwAsCaller(exception);
end
end

function iValidateDispatchersHaveSameNumberOfObservations(dispatchers)
numObservations = dispatchers{1}.NumObservations;
for i = 2:numel(dispatchers)
    if(numObservations == dispatchers{i}.NumObservations)
    else
        exception = iCreateExceptionFromErrorID('nnet_cnn:DAGNetwork:NumObservationsMismatch');
        throwAsCaller(exception);
    end
end
end

function iValidateDispatchersForActivations(dispatchers, inputSizes)
numInputLayers = numel(inputSizes);
for i = 1:numInputLayers
    iAssertDispatcherIsValidForActivations(dispatchers{i}, inputSizes{i});
end
end

function dispatcherInputSizes = iGetDispatcherInputSizes(dispatchers)
numDispatchers = numel(dispatchers);
dispatcherInputSizes = cell(1,numDispatchers);
for i = 1:numDispatchers
    dispatcherInputSizes{i} = iInputSize(dispatchers{i});
end
end

function iAssertDispatcherIsValidForActivations( dispatcher, inputSize )
% iAssertDispatcherIsValidForActivations   Throws an error if the
% dispatcher doesn't give input that will work with the 'activations'
% method.

if iDispatchesImagesSmallerThan(dispatcher, inputSize)
    throwAsCaller( iCreateExceptionFromErrorID( 'nnet_cnn:SeriesNetwork:SmallerImagesForActivations', ...
        mat2str( inputSize ) ) );
end
end

function tf = iDispatchesImagesSmallerThan(dispatcher, inputSize)
% iDispatchesImagesSmallerThan   Check if the dispatcher dispatches images
% smaller than inputSize on either one of the first two dimensions or with
% different number of channels.
dispatchedImageSize = iInputSize( dispatcher );
tf = dispatchedImageSize(1) < inputSize(1) || ...
    dispatchedImageSize(2) < inputSize(2) || ...
    dispatchedImageSize(3) ~= inputSize(3);
end

function sz = iInputSize( x )
% iInputSize   Return the size of x as [H W C] where C is 1 when x is a
% grayscale image.
if iIsDataDispatcher( x )
    sz = x.ImageSize;
    if isempty(sz) && isprop(x, 'DataSize')
        sz = x.DataSize;
    end
else
    sz = [size(x,1) size(x,2) size(x,3)];
end
end

function tf = iIsDataDispatcher(x)
tf = isa(x,'nnet.internal.cnn.DataDispatcher');
end

function X = iWrapInCell(X)
if ~iscell(X)
    X = {X};
end
end

function dispatchers = iStartDispatchers(dispatchers)
for i = 1:numel(dispatchers)
    dispatchers{i}.start();
end
end

function [X, indices] = iNextMiniBatch(dispatchers)
numDispatchers = numel(dispatchers);
X = cell(1, numDispatchers);
[X{1}, ~, indices] = dispatchers{1}.next();
for i = 2:numDispatchers
    X{i} = dispatchers{i}.next();
end
end

function YFormatted = iFormatPredictionsAs2DRowResponses(Y)
% iFormatPredictionsAs2DRowResponses   Format predictions according to the
% problem. 
% If Y is [1 1 K N], then YFormatted will be [N K]. 
% If Y is [H W K N], with H or W not singleton, then YFormatted will be the
% same as Y.

YSize = size(Y);
if YSize(1)==1 && YSize(2)==1
    % Get rid of first two singleton dimensions
    YFormatted = shiftdim(Y,2);
    % Transpose [K N] -> [N K]
    YFormatted = YFormatted';
else
    YFormatted = Y;
end
end

function tf = iIsClassificationOutputLayer(internalLayer)
tf = isa(internalLayer, 'nnet.internal.cnn.layer.ClassificationLayer');
end

function labels = iUndummify(scores, classNames)
labels = nnet.internal.cnn.util.undummify( scores, classNames );
end