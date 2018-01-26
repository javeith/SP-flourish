classdef DAGNetwork < nnet.internal.cnn.TrainableNetwork
    % DAGNetwork   Class for a directed acyclic graph network
    %
    %   A DAG Network is the most general form of
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % Topologically sorted layers
        Layers
        
        % Topologically sorted connections
        Connections
    end
    
    properties
        % NumInputLayers   The number of input layers for this network
        %   The number of input layers for this network. This property is
        %   public because it is needed by the other DAGNetwork object.
        NumInputLayers
        
        % NumOutputLayers   The number of output layers for this network
        %   The number of output layers for this network. This property is
        %   public because it is needed by the other DAGNetwork object.
        NumOutputLayers
        
        % InputLayerIndices   The indices of the input layers
        %   The indices of the input layers.
        InputLayerIndices
        
        % OutputLayerIndices   The indices of the output layers
        %   The indices of the output layers.
        OutputLayerIndices
        
        % InputSizes   Sizes of the network inputs
        InputSizes
        
        % Outputsizes   Sizes of the network outputs
        OutputSizes
        
        % TopologicalOrder  Topological order of layers in the OriginalLayers array
        TopologicalOrder
    end
    
    properties(Access = private)
        % NumActivations   Number of activations
        %   The number of unique output activations in the network.
        NumActivations
        
        % ListOfBufferInputIndices   List of buffer input indices
        %   When we do forward propagation for a graph, we store
        %   activations in a linear buffer. This property is a list, with 
        %   one entry for each layer. Each entry is a vector of the indices
        %   in the linear buffer that are the inputs to this layer. For
        %   example, if the 4th entry stores the vector [1 2], then that
        %   means the 1st and 2nd entries of the linear buffer are the
        %   inputs to the 4th layer.
        ListOfBufferInputIndices
        
        % ListOfBufferOutputIndices   List of buffer output indices
        %   When we do forward propagation for a graph, we store
        %   activations in a linear buffer. This property is a list, with
        %   one entry for each layer. Each entry is a vector of the indices
        %   in the linear buffer that are outputs from this layer. For
        %   example, if the 2nd entry stores the vector [3 4], then that
        %   means the 3rd and 4th entries of the linear buffer are the
        %   outputs from the 2nd layer.
        ListOfBufferOutputIndices
        
        % EdgeTable
        EdgeTable
        
        % Sizes   The output sizes for each activation
        Sizes
        
        % LayerOutputSizes  The output sizes for each layer
        LayerOutputSizes
    end
    
    properties(Dependent, Access = private)
        % NumLayers
        NumLayers
    end
    
    properties (Dependent, SetAccess = private)
        % LearnableParameters    Learnable parameters of the networks
        %                        (vector of nnet.internal.cnn.layer.LearnableParameter)
        %   This is needed for training!!!!
        LearnableParameters
        
        % LayerGraph    A layer graph
        %   This contains an internal layer graph with the most recent
        %   learnable parameters and is created using the Layers and
        %   Connections properties.
        LayerGraph
        
        % OriginalLayers  Layers in the original order
        OriginalLayers
        
        % OriginalConnections  Connections in the original order
        OriginalConnections
    end
    
    methods
        function learnableParameters = get.LearnableParameters(this)
            learnableParameters = [];
            for el = 1:this.NumLayers
                thisParam = this.Layers{el}.LearnableParameters;
                if ~isempty( thisParam )
                    learnableParameters = [learnableParameters thisParam]; %#ok<AGROW>
                end
            end
        end
        
        function layerGraph = get.LayerGraph(this)
            layerGraph = makeTrainedLayerGraph(this);
        end
        
        function originalLayers = get.OriginalLayers(this)
            originalLayers = nnet.internal.cnn.LayerGraph.sortedToOriginalLayers(this.Layers, this.TopologicalOrder);
        end
        
        function originalConnections = get.OriginalConnections(this)
            originalConnections = nnet.internal.cnn.LayerGraph.sortedToOriginalConnections(this.Connections, this.TopologicalOrder);
            originalConnections = sortrows(originalConnections);
        end
    end
    
    methods
        function val = get.NumLayers(this)
            val = numel(this.Layers);
        end
    end
    
    methods
        function this = DAGNetwork(sortedLayerGraph, topologicalOrder)
            %DAGNetwork - Create an internal DAGNetwork.
            %   this = DAGNetwork(sortedLayerGraph, topologicalOrder)
            %   creates an internal DAGNetwork. Input sortedLayerGraph is
            %   an internal LayerGraph containing a topologically sorted
            %   array of internal layers and input topologicalOrder is a
            %   vector representing the indices of the sorted internal
            %   layers in the original (unsorted) array of internal layers.
            
            % Get sorted internal layers based on topological order
            sortedGraph = getAugmentedDigraph(sortedLayerGraph);
            sortedInternalLayers = sortedLayerGraph.Layers;
            
            this.Layers = sortedInternalLayers;
            
            % Create an edgeTable with variables EndNodes and EndPorts.
            % Here's an example of what edgeTable should look like:
            %
            % edgeTable =
            %
            %   6×2 table
            %
            %     EndNodes      EndPorts
            %     ________    ____________
            %
            %     1    2      [1×2 double]
            %     2    3      [1×2 double]
            %     3    4      [1×2 double]
            %     4    5      [1×2 double]
            %     5    6      [1×2 double]
            %     6    7      [1×2 double]
            sourceLayer = findnode(sortedGraph,sortedGraph.Edges.EndNodes(:,1));
            targetLayer = findnode(sortedGraph,sortedGraph.Edges.EndNodes(:,2));
            endPorts = sortedGraph.Edges.AllEndPorts;
            edgeTable = table([sourceLayer, targetLayer],endPorts,'VariableNames',{'EndNodes','EndPorts'});
            this.EdgeTable = edgeTable;
            
            this.NumInputLayers = iCountInputLayers(sortedInternalLayers);
            this.NumOutputLayers = iCountOutputLayers(sortedInternalLayers);
            this.InputLayerIndices = iGetInputLayerIndices(sortedInternalLayers);
            this.OutputLayerIndices = iGetOutputLayerIndices(sortedInternalLayers);
            
            % TODO: Right now, we are assuming that input layers only
            % produce one output, and output layers only take one input.
            this = inferSizes(this);
            this.InputSizes = iGetInputSizes(this.LayerOutputSizes, ...
                this.InputLayerIndices);
            this.OutputSizes = iGetOutputSizes(this.LayerOutputSizes, ...
                this.OutputLayerIndices);
            
            this.NumActivations = iGetNumActivations( ...
                sortedInternalLayers, ...
                edgeTable);
            
            this.ListOfBufferOutputIndices = iGenerateListOfBufferOutputIndices( ...
                sortedInternalLayers, ...
                edgeTable);
            
            this.ListOfBufferInputIndices = iGenerateListOfBufferInputIndices( ...
                sortedInternalLayers, ...
                edgeTable, ...
                this.ListOfBufferOutputIndices);
            
            % Save the internal connections. A layer graph with the most
            % recent values of learnable parameters can be accessed using
            % the LayerGraph property.
            this.Connections = iExternalToInternalConnections(this.EdgeTable);
            
            % Save the original layer indices.
            this.TopologicalOrder = topologicalOrder;
        end
        
        function activationsBuffer = forwardPropagationWithPredict(this, X, layerIndex)
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Assume layers are topologically sorted. We will forward
            % propagate from layer 1 to layerIndex. When layerIndex is not
            % supplied, we forward propagate across the entire network.
            if ( nargin < 3 )
                layerIndex = this.NumLayers;
            end
            
            % Allocate space for the activations.
            activationsBuffer = cell(this.NumActivations,1);
            
            for i = 1:layerIndex
                % TODO: We should refactor everything so that calling
                % "forward" on an input layer will just dispatch data. Then
                % we could remove the "if" statement here.
                
                if isa(this.Layers{i},'nnet.internal.cnn.layer.ImageInput')
                        [~, currentInputLayer] = find(this.InputLayerIndices == i);
                    
                        outputActivations = this.Layers{i}.predict(X{currentInputLayer});
                else
                        XForThisLayer = iGetTheseActivationsFromBuffer( ...
                            activationsBuffer, ...
                            this.ListOfBufferInputIndices{i});
                    
                        outputActivations = this.Layers{i}.predict(XForThisLayer);
                end
                
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    this.ListOfBufferOutputIndices{i}, ...
                    outputActivations);
                
                % 'In-place' ReLU. The input to a ReLU can be set equal to
                % its output without loss of accuracy. By making a copy
                % (which will share the same GPU memory) we can retrieve
                % the memory used by that array.
                if isa(this.Layers{i}, 'nnet.internal.cnn.layer.ReLU')
                    activationsBuffer{this.ListOfBufferInputIndices{i}} = ...
                        activationsBuffer{this.ListOfBufferOutputIndices{i}};
                end
            end
        end
        
        function [activationsBuffer, memoryBuffer] = forwardPropagationWithMemory(this, X)
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Allocate space for the activations.
            activationsBuffer = cell(this.NumActivations,1);
            memoryBuffer = cell(this.NumActivations,1);
            
            for i = 1:this.NumLayers
                % TODO: We should refactor everything so that calling
                % "forward" on an input layer will just dispatch data. Then
                % we could remove the "if" statement here.
                
                if isa(this.Layers{i},'nnet.internal.cnn.layer.ImageInput')
                        [~, currentInputLayer] = find(this.InputLayerIndices == i);
                    
                        [outputActivations, memory] = this.Layers{i}.forward(X{currentInputLayer});
                else
                        XForThisLayer = iGetTheseActivationsFromBuffer( ...
                            activationsBuffer, ...
                            this.ListOfBufferInputIndices{i});
                    
                        [outputActivations, memory] = this.Layers{i}.forward(XForThisLayer);
                end
                
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    this.ListOfBufferOutputIndices{i}, ...
                    outputActivations);
                
                memoryBuffer = iAssignMemoryToBuffer( ...
                    memoryBuffer, ...
                    this.ListOfBufferOutputIndices{i}, ...
                    memory);
                
                % 'In-place' ReLU. The input to a ReLU can be set equal to
                % its output without loss of accuracy. By making a copy
                % (which will share the same GPU memory) we can retrieve
                % the memory used by that array.
                if isa(this.Layers{i}, 'nnet.internal.cnn.layer.ReLU')
                    activationsBuffer{this.ListOfBufferInputIndices{i}} = ...
                        activationsBuffer{this.ListOfBufferOutputIndices{i}};
                end
            end
        end
        
        function Y = predict(this, X)
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Assume the layers have been topologically sorted, and that we
            % have all the info we need. Iterate through the layers and do
            % forward propagation. Try and do it in a memory efficient way.
            
            % First layer is special. Has no input.
            
            % To process subsequent layers, we need the input indices.
            
            X = this.applyTransformsForInputLayers(X);            

            % TODO: Do this in a more memory efficient way. We don't need
            % to calculate every single activation.
            activationsBuffer = this.forwardPropagationWithPredict(X);

            % Extract outputs corresponding to output layers and return
            % them.
            Y = cell(1, this.NumOutputLayers);
            for i = 1:this.NumOutputLayers
                % TODO: This code is based on the assumption that an output
                % layer can only ever produce ONE output. Is that OK?
                outputLayerBufferIndex = this.ListOfBufferOutputIndices{this.OutputLayerIndices(i)};
                Y{i} = activationsBuffer{outputLayerBufferIndex};
            end
        end
        
        function Y = activations(this, X, layerIndex)
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Apply transforms for input layers
            X = this.applyTransformsForInputLayers(X);
            
            % Assume the layers have been topologically sorted, and that we
            % have all the info we need. Iterate through the layers and do
            % forward propagation upto layerIndex
            activationsBuffer = this.forwardPropagationWithPredict(X, layerIndex);
            
            % Extract outputs corresponding to layerIndex and return them
            activationBufferIndicesForLayer = this.ListOfBufferOutputIndices{layerIndex};
            numActivations = numel(activationBufferIndicesForLayer);
            Y = cell(1,numActivations);
            for i = 1:numActivations
                Y{i} = activationsBuffer{activationBufferIndicesForLayer(i)};
            end
        end
        
        function [gradients, predictions] = computeGradientsForTraining(this, X, Y)
            % Wrap X and Y in cell if needed
            X = iWrapInCell(X);
            Y = iWrapInCell(Y);
            
            % Do forward and get all activations
            [activationsBuffer, memoryBuffer] = this.forwardPropagationWithMemory(X);
            
            % Compute the backward loss for the loss layers.
            %dLossdZ = this.Layers{}.backward(layerOutputs{},Y)
            
            % Do backward and delete activations as we go. We delete the
            % outputs for each layer as we go.
            dLossdXBuffer = cell(this.NumActivations,1);
            gradients = {};
            for i = this.NumLayers:-1:1                
                if isa(this.Layers{i}, 'nnet.internal.cnn.layer.OutputLayer')
                    % Perform backpropagation for an output layer
                    ZForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, ...
                        this.ListOfBufferOutputIndices{i});
                    [~, currentInputLayer] = find(this.OutputLayerIndices == i);
                    TForThisLayer = Y{currentInputLayer};
                    
                    dLossdX = this.Layers{i}.backwardLoss( ...
                        ZForThisLayer, TForThisLayer);
                    
                    dLossdXBuffer = iIncrementActivationsInBuffer( ...
                        dLossdXBuffer, ...
                        this.ListOfBufferInputIndices{i}, ...
                        dLossdX);
                elseif isa(this.Layers{i}, 'nnet.internal.cnn.layer.ImageInput')
                    % Do nothing
                else
                    % Perform backpropagation for some other kind of
                    % layer
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, ...
                        this.ListOfBufferInputIndices{i});
                    ZForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, ...
                        this.ListOfBufferOutputIndices{i});
                    dLossdZ = iGetTheseActivationsFromBuffer( ...
                        dLossdXBuffer, ...
                        this.ListOfBufferOutputIndices{i});
                    memory = iGetTheseActivationsFromBuffer( ...
                        memoryBuffer, ...
                        this.ListOfBufferOutputIndices{i});
                    
                    [dLossdX, dLossdW] = this.Layers{i}.backward( ...
                        XForThisLayer, ...
                        ZForThisLayer, ...
                        dLossdZ, ...
                        memory);
                    
                    dLossdXBuffer = iIncrementActivationsInBuffer( ...
                        dLossdXBuffer, ...
                        this.ListOfBufferInputIndices{i}, ...
                        dLossdX);
                    
                    gradients = [dLossdW gradients]; %#ok<AGROW>
                end
            end
            
            % TODO: This code has been copied from "predict".
            predictions = cell(1, this.NumOutputLayers);
            for i = 1:this.NumOutputLayers
                outputLayerBufferIndex = this.ListOfBufferOutputIndices{this.OutputLayerIndices(i)};
                predictions{i} = activationsBuffer{outputLayerBufferIndex};
            end
        end
        
        function loss = loss(this, Y, T)
            % Wrap Y and T in cell if needed
            Y = iWrapInCell(Y);
            T = iWrapInCell(T);
            
            % loss   Calculate the network loss
            loss = [];
            for i = 1:this.NumOutputLayers
                loss = [loss this.Layers{this.OutputLayerIndices(i)}.forwardLoss(Y{i}, T{i})]; %#ok<AGROW>
            end
            loss = sum(loss);
        end
        
        function this = updateLearnableParameters(this, deltas)
            % updateLearnableParameters   Update each learnable parameter
            % by subtracting a delta from it
            currentDelta = 1;
            for el = 1:this.NumLayers
                for param = 1:numel(this.Layers{el}.LearnableParameters)
                    this.Layers{el}.LearnableParameters(param).Value = this.Layers{el}.LearnableParameters(param).Value + deltas{currentDelta};
                    currentDelta = currentDelta + 1;
                end
            end
        end
        
        function this = initializeLearnableParameters(this, precision)
            % initializeLearnableParameters   Initialize the learnable
            % parameters of the network
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.initializeLearnableParameters(precision);
            end
        end
        
        function this = prepareNetworkForTraining(this, executionSettings)
            % prepareNetworkForTraining   Convert the network into a format
            % suitable for training
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.prepareForTraining();
            end
            
            % Determine whether training should occur on host or GPU
            if ismember( executionSettings.executionEnvironment, {'gpu'} )
                % Don't move data if training in parallel, allow this to
                % happen as training progresses. This ensures we can
                % support clients without GPUs when the cluster has GPUs.
                delayMove = executionSettings.useParallel;
                this = this.setupNetworkForGPUTraining(delayMove);
            else
                this = this.setupNetworkForHostTraining();
            end
        end
        
        function this = prepareNetworkForPrediction(this)
            % prepareNetworkForPrediction   Convert the network into a 
            % format suitable for prediction
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.prepareForPrediction();
            end
        end
        
        function this = setupNetworkForHostPrediction(this)
            % setupNetworkForHostPrediction   Setup the network to perform
            % prediction on the host
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.setupForHostPrediction();
            end
        end
        
        function this = setupNetworkForGPUPrediction(this)
            % setupNetworkForGPUPrediction   Setup the network to perform
            % prediction on the GPU
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.setupForGPUPrediction();
            end
        end
        
        function this = setupNetworkForHostTraining(this)
            % setupNetworkForHostTraining   Setup the network to train on
            % the host
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.setupForHostTraining();
                this.Layers{el} = this.Layers{el}.moveToHost();
            end
        end
        
        function this = setupNetworkForGPUTraining(this, deferMove)
           % setupNetworkForGPUTraining   Setup the network to train on
           % the GPU. deferMove allows the actual move of data to the GPU
           % to be deferred to happen as training progresses instead of in
           % advance.
           for el = 1:this.NumLayers
              this.Layers{el} = this.Layers{el}.setupForGPUTraining();
              if ~deferMove
                  this.Layers{el} = this.Layers{el}.moveToGPU();
              end
           end
        end
        
        function indices = namesToIndices(this, stringArray)
            % namesToIndices   Convert a string array of layer names into
            % layer indices
            numLayersToMatch = numel(stringArray);
            indices = zeros(numLayersToMatch,1);
            layerNames = nnet.internal.cnn.layer.Layer.getLayerNames(this.Layers);
            for i = 1:numLayersToMatch
                indices(i) = find(strcmp(stringArray(i), layerNames));
            end
        end
        
         function this = finalizeNetwork(this, X)
            % Wrap X in cell if needed
            X = iWrapInCell(X);
             
            % finalizeNetwork
            
            activationsBuffer = cell(this.NumActivations,1);
          
           % Allocate space for the activations.
            
            for i = 1:this.NumLayers
                % TODO: We should refactor everything so that calling
                % "forward" on an input layer will just dispatch data. Then
                % we could remove the "if" statement here.
                
                layerType = class(this.Layers{i});
                switch layerType
                    case 'nnet.internal.cnn.layer.ImageInput'
                        [~, currentInputLayer] = find(this.InputLayerIndices == i);
                    
                        [Z, memory] = this.Layers{i}.forward(X{currentInputLayer});
                    otherwise
                        XForThisLayer = iGetTheseActivationsFromBuffer( ...
                            activationsBuffer, ...
                            this.ListOfBufferInputIndices{i});
                    
                        [Z, memory] = this.Layers{i}.forward(XForThisLayer);
                end
                
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    this.ListOfBufferOutputIndices{i}, ...
                    Z);
                
                if  isa(this.Layers{i},'nnet.internal.cnn.layer.Finalizable')
                    this.Layers{i} = finalize(this.Layers{i}, XForThisLayer, Z, memory);
                end
                
                            
            end           

         end
        
         function this = inferSizes(this)
             % inferSizes   Infer layer output sizes
             
             sortedInternalLayers = this.Layers;
             edgeTable = this.EdgeTable;
             
             numActivations = iGetNumActivations( ...
                 sortedInternalLayers, ...
                 edgeTable);
             
             listOfBufferOutputIndices = iGenerateListOfBufferOutputIndices( ...
                 sortedInternalLayers, ...
                 edgeTable);
             
             listOfBufferInputIndices = iGenerateListOfBufferInputIndices( ...
                 sortedInternalLayers, ...
                 edgeTable, ...
                 listOfBufferOutputIndices);
             
             this.Sizes = cell(numActivations,1);
             numLayers = numel(sortedInternalLayers);
             this.LayerOutputSizes = cell(numLayers,1);
             
             for i = 1:numLayers
                 if isa(sortedInternalLayers{i}, 'nnet.internal.cnn.layer.ImageInput')
                     inputSizesForThisLayer = sortedInternalLayers{i}.InputSize;
                 else
                     inputSizesForThisLayer = iGetInputsFromBuffer( ...
                         this.Sizes, listOfBufferInputIndices{i});
                 end
                 
                 sortedInternalLayers{i} = iInferSize( ...
                     sortedInternalLayers{i}, ...
                     inputSizesForThisLayer, ...
                     i);
                 
                 outputSizesForThisLayer = sortedInternalLayers{i}.forwardPropagateSize( ...
                     inputSizesForThisLayer);
                 this.Sizes = iAssignOutputsToBuffer( ...
                     this.Sizes, listOfBufferOutputIndices{i}, outputSizesForThisLayer);
                 this.LayerOutputSizes{i} = outputSizesForThisLayer;
             end
         end
         
         function layerOutputSizes = inferOutputSizesGivenInputSizes(this, inputSizes)
             % inferOutputSizesGivenInputSizes   Infer layer output sizes
             % given new input sizes for input layers.
             %
             % Suppose this internal DAG network has N layers which have
             % been topologically sorted and numbered from 1 to N. Suppose
             % the network has M input layers and they appear in positions
             % i_1, i_2, ..., i_M in the topologically sorted list.
             %
             % inputSizes       - is a length M cell array specifying the
             %                    input sizes for layers i_1, i_2, ..., i_M
             %                    in that order.
             %
             % layerOutputSizes - is a length N cell array such that
             %                    layerOutputSizes{i} is the output size
             %                    for layer i. If layer i has multiple
             %                    outputs then layerOutputSizes{i} is a
             %                    cell array of output sizes for layer i.
             
             sortedInternalLayers = this.Layers;
             edgeTable = this.EdgeTable;
             
             numActivations = iGetNumActivations( ...
                 sortedInternalLayers, ...
                 edgeTable);
             
             listOfBufferOutputIndices = iGenerateListOfBufferOutputIndices( ...
                 sortedInternalLayers, ...
                 edgeTable);
             
             listOfBufferInputIndices = iGenerateListOfBufferInputIndices( ...
                 sortedInternalLayers, ...
                 edgeTable, ...
                 listOfBufferOutputIndices);
             
             sizes = cell(numActivations,1);
             numLayers = numel(sortedInternalLayers);
             layerOutputSizes = cell(numLayers,1);
             
             for i = 1:numLayers
                 if isa(sortedInternalLayers{i}, 'nnet.internal.cnn.layer.ImageInput')
                     % For an image input layer, forwardPropagateSize sets
                     % the output size equal to the InputSize property.
                     % Since we don't want that, we force the output size
                     % to be equal to the specified input size.
                     [~, currentInputLayer] = find(this.InputLayerIndices == i);
                     inputSizesForThisLayer = inputSizes{currentInputLayer};
                     outputSizesForThisLayer = inputSizesForThisLayer;
                 else
                     inputSizesForThisLayer = iGetInputsFromBuffer( ...
                         sizes, listOfBufferInputIndices{i});
                     outputSizesForThisLayer = sortedInternalLayers{i}.forwardPropagateSize( ...
                         inputSizesForThisLayer);
                 end
                 
                 sizes = iAssignOutputsToBuffer( ...
                     sizes, listOfBufferOutputIndices{i}, outputSizesForThisLayer);
                 layerOutputSizes{i} = outputSizesForThisLayer;
             end
         end
         
         function layerGraph = makeTrainedLayerGraph(this)
             % makeTrainedLayerGraph - makes an internal Layer graph
             % with most recent values of learnable parameters
             layerGraph = iMakeInternalLayerGraph(this.OriginalLayers, this.OriginalConnections);
         end
    end
    
    methods(Access = private)
        function X = applyTransformsForInputLayers(this, X)
            % TODO: All this stuff should be done when you call "forward"
            % on the input layers.
            numInputLayers = numel(X);
            for i = 1:numInputLayers
                currentLayer = this.InputLayerIndices(i);
                X{i} = apply(this.Layers{currentLayer}.Transforms, X{i});
            end
        end
    end
end

function layerGraph = iMakeInternalLayerGraph(layers, connections)
layerGraph = nnet.internal.cnn.LayerGraph(layers, connections);
end

function internalConnections = iExternalToInternalConnections( externalConnections )
externalEndNodes = externalConnections.EndNodes;
externalEndPorts = externalConnections.EndPorts;
numEndPortsPerEndNodes = cellfun(@(x) size(x,1), externalEndPorts);
internalEndPorts = cell2mat(externalEndPorts);
internalEndNodes = [repelem(externalEndNodes(:,1),numEndPortsPerEndNodes), repelem(externalEndNodes(:,2),numEndPortsPerEndNodes)];
internalConnections = [internalEndNodes(:,1),internalEndPorts(:,1),internalEndNodes(:,2),internalEndPorts(:,2)];
end

function X = iWrapInCell(X)
if ~iscell(X)
    X = {X};
end
end

function numInputLayers = iCountInputLayers(internalLayers)
numInputLayers = 0;
for i = 1:numel(internalLayers)
    if( iIsAnInputLayer(internalLayers{i}) )
        numInputLayers = numInputLayers + 1;
    end
end
end

function numOutputLayers = iCountOutputLayers(internalLayers)
numOutputLayers = 0;
for i = 1:numel(internalLayers)
    if( iIsAnOutputLayer(internalLayers{i}) )
        numOutputLayers = numOutputLayers + 1;
    end
end
end

function inputLayerIndices = iGetInputLayerIndices(internalLayers)
numLayers = numel(internalLayers);
inputLayerIndices = cell(1,numLayers);
for i = 1:numLayers
    if(iIsAnInputLayer(internalLayers{i}))
        inputLayerIndices{i} = i;
    end
end
inputLayerIndices = cat(2,inputLayerIndices{:});
end

function outputLayerIndices = iGetOutputLayerIndices(internalLayers)
numLayers = numel(internalLayers);
outputLayerIndices = cell(1,numLayers);
for i = 1:numLayers
    if(iIsAnOutputLayer(internalLayers{i}))
        outputLayerIndices{i} = i;
    end
end
outputLayerIndices = cat(2, outputLayerIndices{:});
end

function inputSizes = iGetInputSizes(sizes, inputLayerIndices)
numInputLayers = numel(inputLayerIndices);
inputSizes = cell(1, numInputLayers);
for i = 1:numInputLayers
    currentLayer = inputLayerIndices(i);
    inputSizes{i} = sizes{currentLayer};
end
end

function outputSizes = iGetOutputSizes(sizes, outputLayerIndices)
numOutputLayers = numel(outputLayerIndices);
outputSizes = cell(1, numOutputLayers);
for i = 1:numOutputLayers
    currentLayer = outputLayerIndices(i);
    outputSizes{i} = sizes{currentLayer};
end
end

function tf = iIsAnInputLayer(internalLayer)
% TODO: We should really have an inheritance hierarchy for input layers.
% All input layers should inherit from some kind of abstract input layer.
tf = isa(internalLayer,'nnet.internal.cnn.layer.ImageInput');
end

function tf = iIsAnOutputLayer(internalLayer)
tf = isa(internalLayer,'nnet.internal.cnn.layer.OutputLayer');
end

%--------------------------------------------------------------------------
% BEGIN STUFF RELATED TO PORTS
% The stuff below should be pulled into a class somehow
%--------------------------------------------------------------------------

function numActivations = iGetNumActivations(internalLayers, edgeTable)
numActivations = 0;
for i = 1:numel(internalLayers)
    % TODO: Each layer should know how many outputs it has.
    numOutputsForThisLayer = nnet.internal.cnn.layer.util.getNumOutputs(internalLayers{i});
    if(isnan(numOutputsForThisLayer))
        numOutputsForThisLayer = countNumberOfUniqueOutputsFromLayer(edgeTable, i);
    end
    numActivations = numActivations + numOutputsForThisLayer;
end
end

function listOfBufferOutputIndices = iGenerateListOfBufferOutputIndices(sortedInternalLayers, edgeTable)
numLayers = numel(sortedInternalLayers);
listOfBufferOutputIndices = cell(numLayers, 1);
offset = 0;
for i = 1:numLayers
    numOutputsForThisLayer = nnet.internal.cnn.layer.util.getNumOutputs( ...
        sortedInternalLayers{i});
    if(isnan(numOutputsForThisLayer))
        % This layer has a variable number of outputs, so we need to count them.
        numOutputsForThisLayer = countNumberOfUniqueOutputsFromLayer(edgeTable, i);
    end
    listOfBufferOutputIndices{i} = (1:numOutputsForThisLayer) + offset;
    offset = offset + numOutputsForThisLayer;
end
end

function numUniqueOutputs = countNumberOfUniqueOutputsFromLayer(edgeTable, layerIndex)
% Get the connections coming out of this layer
edgesOutOfLayerTable = edgeTable((edgeTable.EndNodes(:,1) == layerIndex), {'EndNodes','EndPorts'});

% Count the number of unique ports coming out of this layer
numUniqueOutputs = countNumberOfUniqueOutputPorts(edgesOutOfLayerTable);
end

function numUniqueOutputPorts = countNumberOfUniqueOutputPorts(edgesOutOfLayerTable)
portList = cell2mat(edgesOutOfLayerTable.EndPorts);
startPortList = portList(:,1);
numUniqueOutputPorts = numel(unique(startPortList));
end

function listOfBufferInputIndices = iGenerateListOfBufferInputIndices( ...
    sortedInternalLayers, edgeTable, listOfBufferOutputIndices)

numLayers = numel(sortedInternalLayers);
listOfBufferInputIndices = cell(numLayers,1);

for i = 1:numLayers
    % Get the connections feeding into this layer
    edgesIntoLayerTable = edgeTable((edgeTable.EndNodes(:,2) == i), {'EndNodes','EndPorts'});

    % Map inputs for each layer to indices in the activations buffer
    listOfBufferInputIndices{i} = iMapInputPortsToBufferIndices(edgesIntoLayerTable, listOfBufferOutputIndices);
end

end

function listOfBufferInputIndices = iMapInputPortsToBufferIndices(edgesIntoLayerTable, listOfBufferOutputIndices)

edgeMatrix = iConvertEdgeTableToMatrix(edgesIntoLayerTable);
edgeMatrix = iSortEdgeMatrixByInputPort(edgeMatrix);
layerIndexList = edgeMatrix(:,1);
outputPortList = edgeMatrix(:,3);

% Map to buffer
firstBufferOutputIndices = iGetFirstBufferOutputIndices(listOfBufferOutputIndices);
listOfBufferInputIndices = firstBufferOutputIndices(layerIndexList) + outputPortList - 1;
listOfBufferInputIndices = listOfBufferInputIndices';

end

function outputMatrix = iSortEdgeMatrixByInputPort(inputMatrix)
[~, sortedIndices] = sort(inputMatrix(:,4));
outputMatrix = inputMatrix(sortedIndices,:);
end

function edgeMatrix = iConvertEdgeTableToMatrix(edgeTable)
% iConvertEdgeTableToMatrix
%   The table of edges can be difficult to work with, because
portMatrix = cell2mat(edgeTable.EndPorts);
numUniqueEdges = size(portMatrix,1);
edgeMatrix = zeros(numUniqueEdges, 4);
edgeMatrix(:,3:4) = portMatrix;
edgeMatrix(:,1:2) = iExpandLayerMatrix(edgeTable, numUniqueEdges);
end

function expandedLayerMatrix = iExpandLayerMatrix(edgeTable, numUniqueEdges)
expandedLayerMatrix = zeros(numUniqueEdges, 2);
numNonUniqueEdges = height(edgeTable);
startIndex = 1;
for i = 1:numNonUniqueEdges
    % Each non-unique edge may correspond to several unique edges, because we have multiple ports.
    portList = edgeTable.EndPorts{i};
    numPortEdges = size(portList,1);
    expandedEdgeMatrix = repmat(edgeTable.EndNodes(i,:), [numPortEdges 1]);
    stopIndex = startIndex + numPortEdges - 1;
    expandedLayerMatrix(startIndex:stopIndex,:) = expandedEdgeMatrix;
    startIndex = stopIndex + 1;
end
end

function firstBufferOutputIndices = iGetFirstBufferOutputIndices(listOfBufferOutputIndices)
numLayers = numel(listOfBufferOutputIndices);
firstBufferOutputIndices = zeros(numLayers, 1);
for i = 1:numLayers
    firstBufferOutputIndices(i) = listOfBufferOutputIndices{i}(1);
end
end

%--------------------------------------------------------------------------
% END STUFF RELATED TO PORTS
%--------------------------------------------------------------------------

function XForThisLayer = iGetTheseActivationsFromBuffer(activationsBuffer, inputIndices)
% TODO: Need to decide if inputs should be cell arrays in general
XForThisLayer = activationsBuffer(inputIndices);
if(iscell(XForThisLayer) && (numel(XForThisLayer) == 1))
    XForThisLayer = XForThisLayer{1};
end
end

function memoryBuffer = iAssignMemoryToBuffer(...
    memoryBuffer, ...
    bufferIndices, ...
    memory)
% TODO should default Layer.forward return {[], []} for memory when there
% are multiple outputs? Just duplicate for now. FYI Batch norm stores its
% memory as a cell. 
for i = 1:numel(bufferIndices)
    memoryBuffer{bufferIndices(i)} = memory;
end
end

function activationsBuffer = iAssignActivationsToBuffer( ...
    activationsBuffer, ...
    bufferIndices, ...
    activations)

numActivationsFromLayer = numel(bufferIndices);
if ~iscell(activations)
    activationsBuffer{bufferIndices} = activations;
else
    for i = 1:numActivationsFromLayer
        activationsBuffer{bufferIndices(i)} = activations{i};
    end
end

end

function activationsBuffer = iIncrementActivationsInBuffer(activationsBuffer, bufferIndices, activations)

numActivationsFromLayer = numel(bufferIndices);
if ~iscell(activations)
    if isempty(activationsBuffer{bufferIndices})
        activationsBuffer{bufferIndices} = activations;
    else
        activationsBuffer{bufferIndices} = activationsBuffer{bufferIndices} + activations;
    end
else
    for i = 1:numActivationsFromLayer
        if isempty(activationsBuffer{bufferIndices(i)})
            activationsBuffer{bufferIndices(i)} = activations{i};
        else
            activationsBuffer{bufferIndices(i)} = activationsBuffer{bufferIndices(i)}+ activations{i};
        end
    end
end
end

function internalLayer = iInferSize(internalLayer, inputSize, index)
if(~internalLayer.HasSizeDetermined)
    % Infer layer size if its size is not determined
    try
        internalLayer = internalLayer.inferSize(inputSize);
    catch e
        throwWrongLayerSizeException( e, index );
    end
else
    % Otherwise make sure the size of the layer is correct
    iAssertCorrectSize( internalLayer, index, inputSize );
end
end

function activationsBuffer = iAssignOutputsToBuffer( ...
    activationsBuffer, ...
    outputIndices, ...
    outputActivations)

numOutputsFromLayer = numel(outputIndices);
if ~iscell(outputActivations)
    activationsBuffer{outputIndices} = outputActivations;
else
    for i = 1:numOutputsFromLayer
        activationsBuffer{outputIndices(i)} = outputActivations{i}; 
    end
end
end

function iAssertCorrectSize( internalLayer, index, inputSize )
% iAssertCorrectSize   Check that layer size matches the input size,
% otherwise the architecture would be inconsistent.
if ~internalLayer.isValidInputSize( inputSize )
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:WrongLayerSize', index);
    throwAsCaller(exception);
end
end

function throwWrongLayerSizeException(e, index)
% throwWrongLayerSizeException   Throws a getReshapeDims:notSameNumel exception as
% a WrongLayerSize exception
if (strcmp(e.identifier,'MATLAB:getReshapeDims:notSameNumel'))
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:WrongLayerSize', index);
    throwAsCaller(exception)
else
    rethrow(e)
end
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(message(errorID, varargin{:}));
end

function XForThisLayer = iGetInputsFromBuffer(layerOutputs, inputIndices)
% TODO: Need to decide if inputs should be cell arrays in general
XForThisLayer = layerOutputs(inputIndices);
if(iscell(XForThisLayer) && (numel(XForThisLayer) == 1))
    XForThisLayer = XForThisLayer{1};
end
end