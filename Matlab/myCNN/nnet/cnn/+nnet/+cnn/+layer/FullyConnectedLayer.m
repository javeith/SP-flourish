classdef FullyConnectedLayer < nnet.cnn.layer.Layer
    % FullyConnectedLayer   Fully connected layer
    %
    %   To create a fully connected layer, use fullyConnectedLayer
    %
    %   A fully connected layer. This layer has weight and bias parameters
    %   that are learned during training.
    %
    %   FullyConnectedLayer properties:
    %       Name                        - A name for the layer.
    %       InputSize                   - The input size of the fully
    %                                     connected layer.
    %       OutputSize                  - The output size of the fully
    %                                     connected layer.
    %       Weights                     - The weight matrix.
    %       Bias                        - The bias vector.
    %       WeightLearnRateFactor       - The learning rate factor for the
    %                                     weights.
    %       WeightL2Factor              - The L2 regularization factor for
    %                                     the weights.
    %       BiasLearnRateFactor         - The learning rate factor for the
    %                                     bias.
    %       BiasL2Factor                - The L2 regularization factor for
    %                                     the bias.
    %
    %   Example:
    %       Create a fully connected layer with an output size of 10, and an
    %       input size that will be determined at training time.
    %
    %       layer = fullyConnectedLayer(10);
    %
    %   See also fullyConnectedLayer
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties(SetAccess = private, Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
        
        % InputSize   The input size for the layer
        %   The input size for the fully connected layer. If this is set to
        %   'auto', then the input size will be automatically set at
        %   training time.
        InputSize
        
        % OutputSize   The output size for the layer
        %   The output size for the fully connected layer.
        OutputSize
    end
    
    properties(Dependent)
        % Weights   The weights for the layer
        %   The weight matrix for the fully connected layer. This matrix
        %   will have size OutputSize-by-InputSize.
        Weights
        
        % Bias   The biases for the layer
        %   The bias vector for the fully connected layer. This vector will
        %   have size OutputSize-by-1.
        Bias
        
        % WeightLearnRateFactor   The learning rate factor for the weights
        %   The learning rate factor for the weights. This factor is
        %   multiplied with the global learning rate to determine the
        %   learning rate for the weights in this layer. For example, if it
        %   is set to 2, then the learning rate for the weights in this
        %   layer will be twice the current global learning rate.
        WeightLearnRateFactor
        
        % WeightL2Factor   The L2 regularization factor for the weights
        %   The L2 regularization factor for the weights. This factor is
        %   multiplied with the global L2 regularization setting to
        %   determine the L2 regularization setting for the weights in this
        %   layer. For example, if it is set to 2, then the L2
        %   regularization for the weights in this layer will be twice the
        %   global L2 regularization setting.
        WeightL2Factor
        
        % BiasLearnRateFactor   The learning rate factor for the biases
        %   The learning rate factor for the bias. This factor is
        %   multiplied with the global learning rate to determine the
        %   learning rate for the bias in this layer. For example, if it
        %   is set to 2, then the learning rate for the bias in this layer
        %   will be twice the current global learning rate.
        BiasLearnRateFactor
        
        % BiasL2Factor   The L2 regularization factor for the biases
        %   The L2 regularization factor for the biases. This factor is
        %   multiplied with the global L2 regularization setting to
        %   determine the L2 regularization setting for the biases in this
        %   layer. For example, if it is set to 2, then the L2
        %   regularization for the biases in this layer will be twice the
        %   global L2 regularization setting.
        BiasL2Factor
    end
    
    methods
        function val = get.InputSize(this)
            if ~isempty(this.PrivateLayer.InputSize)
                % Get the input size from the internal 4-D input size.
                val = prod(this.PrivateLayer.InputSize);
            elseif ~isempty(this.PrivateLayer.Weights.Value)
                % If the weights have been set externally as 2-D matrix
                % the user visible size is available. The internal size
                % will be determined when the weights will be reshaped
                % to 4-D.
                val = size(this.PrivateLayer.Weights.Value, 2);
            else
                val = 'auto';
            end
        end
        
        function val = get.OutputSize(this)
            val = this.PrivateLayer.NumNeurons;
        end
        
        function weights = get.Weights(this)
            privateWeights = this.PrivateLayer.Weights.HostValue;
            
            if isempty(privateWeights)
                % If no weights have been defined, return "empty" for
                % weights
                weights = [];
                
            elseif ismatrix(privateWeights)
                % If the weights are in a 2d matrix, then they can just be
                % returned as is
                weights = privateWeights;
                
            else % Default case: 4d array
                % In case the internal weights are 4-D we need to reshape
                % them to 2-D.
                weights = reshape(privateWeights, [], this.OutputSize);
                weights = weights';
            end
        end
        
        function this = set.Weights(this, value)
            classes = {'single', 'double', 'gpuArray'};
            if ~isequal(this.InputSize, 'auto')
                expectedInputSize = prod(this.InputSize);
            else
                expectedInputSize = NaN;
            end
            attributes = {'size', [this.OutputSize expectedInputSize], 'nonempty', 'real', 'nonsparse'};
            validateattributes(value, classes, attributes);
            
            this.PrivateLayer.Weights.Value = gather(value);
        end
        
        function val = get.Bias(this)
            val = this.PrivateLayer.Bias.HostValue;
            if(~isempty(val))
                val = reshape(val, this.OutputSize, 1);
            end
        end
        
        function this = set.Bias(this, value)
            classes = {'single', 'double', 'gpuArray'};
            attributes = {'column', 'nonempty', 'real', 'nonsparse', 'nrows', this.OutputSize};
            validateattributes(value, classes, attributes);
            
            this.PrivateLayer.Bias.Value = gather(value);
        end
        
        function val = get.WeightLearnRateFactor(this)
            val = this.PrivateLayer.Weights.LearnRateFactor;
        end
        
        function this = set.WeightLearnRateFactor(this, value)
            iValidateScalar(value);
            this.PrivateLayer.Weights.LearnRateFactor = value;
        end
        
        function val = get.WeightL2Factor(this)
            val = this.PrivateLayer.Weights.L2Factor;
        end
        
        function this = set.WeightL2Factor(this, value)
            iValidateScalar(value);
            this.PrivateLayer.Weights.L2Factor = value;
        end
        
        function val = get.BiasLearnRateFactor(this)
            val = this.PrivateLayer.Bias.LearnRateFactor;
        end
        
        function this = set.BiasLearnRateFactor(this, value)
            iValidateScalar(value);
            this.PrivateLayer.Bias.LearnRateFactor = value;
        end
        
        function val = get.BiasL2Factor(this)
            val = this.PrivateLayer.Bias.L2Factor;
        end
        
        function this = set.BiasL2Factor(this, value)
            iValidateScalar(value);
            this.PrivateLayer.Bias.L2Factor = value;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
    end
    
    methods(Access = public)
        function this = FullyConnectedLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 2.0;
            out.Name = privateLayer.Name;
            out.InputSize = privateLayer.InputSize;
            out.OutputSize = [1 1 privateLayer.NumNeurons];
            out.Weights = iSaveLearnableParameter(privateLayer.Weights);
            out.Bias = iSaveLearnableParameter(privateLayer.Bias);
        end
    end
    
    methods(Hidden, Static)
        function inputArguments = parseInputArguments(varargin)
            parser = iCreateParser();
            parser.parse(varargin{:});
            inputArguments = iConvertToCanonicalForm(parser);
        end
        
        function this = loadobj(in)
            if in.Version <= 1
                in = iUpgradeVersionOneToVersionTwo(in);
            end
            this = iLoadFullyConnectedLayerFromCurrentVersion(in);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            outputSizeString = int2str( this.OutputSize );
            
            description = iGetMessageString(  ...
                'nnet_cnn:layer:FullyConnectedLayer:oneLineDisplay', ...
                outputSizeString );
            
            type = iGetMessageString( 'nnet_cnn:layer:FullyConnectedLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            hyperparameters = {
                'InputSize'
                'OutputSize'
                };
            
            learnableParameters = {'Weights', 'Bias'};
            
            groups = [
                this.propertyGroupGeneral( {'Name'} )
                this.propertyGroupHyperparameters( hyperparameters )                
                this.propertyGroupLearnableParameters( learnableParameters )                
            ];
        end
        
        function footer = getFooter( this )
            variableName = inputname(1);
            footer = this.createShowAllPropertiesFooter( variableName );
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function S = iSaveLearnableParameter(learnableParameter)
% iSaveLearnableParameter   Save a learnable parameter in the form of a
% structure
S.Value = learnableParameter.Value;
S.LearnRateFactor = learnableParameter.LearnRateFactor;
S.L2Factor = learnableParameter.L2Factor;
end

function S = iUpgradeVersionOneToVersionTwo(S)
% iUpgradeVersionOneToVersionTwo   Upgrade a v1 (2016a) saved struct to a v2 saved struct
%   This means gathering the bias and weights from the GPU and putting them
%   on the host.

S.Version = 2;
try
    S.Bias.Value = gather(S.Bias.Value);
    S.Weights.Value = gather(S.Weights.Value);
catch e
    % Only throw the error we want to throw.
    e = MException( ...
        'nnet_cnn:layer:FullyConnectedLayer:MustHaveGPUToLoadFrom2016a', ...
        getString(message('nnet_cnn:layer:FullyConnectedLayer:MustHaveGPUToLoadFrom2016a')));
    throwAsCaller(e);
end
end

function obj = iLoadFullyConnectedLayerFromCurrentVersion(in)
if ~isempty(in.OutputSize)
    % Remove the first two singleton dimensions of the Outputsize to construct the internal layer.
    in.OutputSize = in.OutputSize(3);
end
internalLayer = nnet.internal.cnn.layer.FullyConnected( ...
    in.Name, in.InputSize, in.OutputSize);
internalLayer.Weights = iLoadLearnableParameter(in.Weights);
internalLayer.Bias = iLoadLearnableParameter(in.Bias);

obj = nnet.cnn.layer.FullyConnectedLayer(internalLayer);
end

function learnableParameter = iLoadLearnableParameter(S)
% iLoadLearnableParameter   Load a learnable parameter from a structure S
learnableParameter = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
learnableParameter.Value = S.Value;
learnableParameter.LearnRateFactor = S.LearnRateFactor;
learnableParameter.L2Factor = S.L2Factor;
end

function p = iCreateParser()
p = inputParser;

defaultWeightLearnRateFactor = 1;
defaultBiasLearnRateFactor = 1;
defaultWeightL2Factor = 1;
defaultBiasL2Factor = 0;
defaultName = '';

p.addRequired('OutputSize', @iIsPositiveIntegerScalar);
p.addParameter('WeightLearnRateFactor', defaultWeightLearnRateFactor, @iIsFiniteRealNumericScalar);
p.addParameter('BiasLearnRateFactor', defaultBiasLearnRateFactor, @iIsFiniteRealNumericScalar);
p.addParameter('WeightL2Factor', defaultWeightL2Factor, @iIsFiniteRealNumericScalar);
p.addParameter('BiasL2Factor', defaultBiasL2Factor, @iIsFiniteRealNumericScalar);
p.addParameter('Name', defaultName, @iIsValidName);
end

function inputArguments = iConvertToCanonicalForm(p)
inputArguments = struct;
inputArguments.InputSize = [];
inputArguments.OutputSize = p.Results.OutputSize;
inputArguments.WeightLearnRateFactor = p.Results.WeightLearnRateFactor;
inputArguments.BiasLearnRateFactor = p.Results.BiasLearnRateFactor;
inputArguments.WeightL2Factor = p.Results.WeightL2Factor;
inputArguments.BiasL2Factor = p.Results.BiasL2Factor;
inputArguments.Name = p.Results.Name;
end

function tf = iIsPositiveIntegerScalar(x)
tf = isscalar(x) && (x > 0) && iIsInteger(x);
end

function tf = iIsFiniteRealNumericScalar(x)
tf = isscalar(x) && isfinite(x) && isreal(x) && isnumeric(x);
end

function tf = iIsInteger(x)
tf = isreal(x) && isnumeric(x) && all(mod(x,1)==0);
end

function tf = iIsValidName(x)
tf = ischar(x);
end

function iValidateScalar(value)
classes = {'numeric'};
attributes = {'scalar', 'nonempty'};
validateattributes(value, classes, attributes);
end