classdef Convolution2DLayer < nnet.cnn.layer.Layer
    % Convolution2DLayer   2-D convolution layer
    %
    %   To create a convolution layer, use convolution2dLayer
    %
    %   Convolution2DLayer properties:
    %       Name                        - A name for the layer.
    %       FilterSize                  - The height and width of the
    %                                     filters.
    %       NumChannels                 - The number of channels for each
    %                                     filter.
    %       NumFilters                  - The number of filters.
    %       Stride                      - The step size for traversing the
    %                                     input vertically and
    %                                     horizontally.
    %       Padding                     - The padding applied to the input
    %                                     vertically and horizontally.
    %       Weights                     - Weights of the layer.
    %       Bias                        - Biases of the layer.
    %       WeightLearnRateFactor       - A number that specifies
    %                                     multiplier for the learning rate
    %                                     of the weights.
    %       BiasLearnRateFactor         - A number that specifies a
    %                                     multiplier for the learning rate
    %                                     for the biases.
    %       WeightL2Factor              - A number that specifies a
    %                                     multiplier for the L2 weight
    %                                     regulariser for the weights.
    %       BiasL2Factor                - A number that specifies a
    %                                     multiplier for the L2 weight
    %                                     regulariser for the biases.
    %
    %   Example:
    %       Create a convolution layer with 5 filters of size 10-by-10.
    %
    %       layer = convolution2dLayer(10, 5);
    %
    %   See also convolution2dLayer.
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties(SetAccess = private, Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
        
        % FilterSize   The height and width of the filters
        %   The height and width of the filters. This is a row vector [h w]
        %   where h is the filter height and w is the filter width.
        FilterSize
        
        % NumChannels   The number of channels in the input
        %   The number of channels in the input. This can be set to 'auto',
        %   in which case the correct value will be determined at training
        %   time.
        NumChannels
        
        % NumFilters   The number of filters
        %   The number of filters for this layer. This also determines how
        %   many maps there will be in the output.
        NumFilters
        
        % Stride   The vertical and horizontal stride
        %   The step size for traversing the input vertically and
        %   horizontally. This is a row vector [u v] where u is the
        %   vertical stride, and v is the horizontal stride.
        Stride
        
        % Padding   The vertical and horizontal padding
        %   The padding that is applied to the input vertically and
        %   horizontally. This a row vector [a b] where a is the padding
        %   applied to the top and bottom of the input, and b is the
        %   padding applied to the left and right of the image.
        Padding
    end
    
    properties(Dependent)
        % Weights   The weights for the layer
        %   The filters for the convolutional layer. An array with size
        %   FilterSize(1)-by-FilterSize(2)-by-NumChannels-by-NumFilters.
        Weights
        
        % Bias   The bias vector for the layer
        %   The bias for the convolutional layer. The size will be
        %   1-by-1-by-NumFilters.
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
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 2.0;
            out.Name = privateLayer.Name;
            out.FilterSize = privateLayer.FilterSize;
            out.NumChannels = privateLayer.NumChannels;
            out.NumFilters = privateLayer.NumFilters;
            out.Stride = privateLayer.Stride;
            out.Padding = privateLayer.Padding;
            out.Weights = iSaveLearnableParameter(privateLayer.Weights);
            out.Bias = iSaveLearnableParameter(privateLayer.Bias);
        end
    end
    
    methods
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function val = get.Weights(this)
            val = this.PrivateLayer.Weights.HostValue;
        end
        
        function this = set.Weights(this, value)
            classes = {'single', 'double', 'gpuArray'};
            if(this.filterGroupsAreUsed())
                expectedNumChannels = iExpectedNumChannels(this.NumChannels(1));
            else
                expectedNumChannels = iExpectedNumChannels(this.NumChannels);
            end
            attributes = {'size', [this.FilterSize expectedNumChannels sum(this.NumFilters)], 'nonempty', 'real'};
            validateattributes(value, classes, attributes);
            
            % Call inferSize to determine the size of the layer
            if(this.filterGroupsAreUsed())
                inputChannels = size(value,3)*2;
            else
                inputChannels = size(value,3);
            end
            this.PrivateLayer = this.PrivateLayer.inferSize( [NaN NaN inputChannels] );
            this.PrivateLayer.Weights.Value = gather(value);
        end
        
        function val = get.Bias(this)
            val = this.PrivateLayer.Bias.HostValue;
        end
        
        function this = set.Bias(this, value)
            classes = {'single', 'double', 'gpuArray'};
            attributes = {'size', [1 1 sum(this.NumFilters)], 'nonempty', 'real'};
            validateattributes(value, classes, attributes);
            
            this.PrivateLayer.Bias.Value = gather(value);
        end
        
        function val = get.FilterSize(this)
            val = this.PrivateLayer.FilterSize;
        end
        
        function val = get.NumChannels(this)
            val = this.PrivateLayer.NumChannels;
            if(this.filterGroupsAreUsed())
                val = [val val];
            end
            if isempty(val)
                val = 'auto';
            end
        end
        
        function val = get.NumFilters(this)
            val = this.PrivateLayer.NumFilters;
        end
        
        function val = get.Stride(this)
            val = this.PrivateLayer.Stride;
        end
        
        function val = get.Padding(this)
            val = this.PrivateLayer.Padding;
        end
        
        function val = get.WeightLearnRateFactor(this)
            val = this.PrivateLayer.Weights.LearnRateFactor;
        end
        
        function this = set.WeightLearnRateFactor(this, value)
            iValidateScalar(value);
            this.PrivateLayer.Weights.LearnRateFactor = value;
        end
        
        function val = get.BiasLearnRateFactor(this)
            val = this.PrivateLayer.Bias.LearnRateFactor;
        end
        
        function this = set.BiasLearnRateFactor(this, value)
            iValidateScalar(value);
            this.PrivateLayer.Bias.LearnRateFactor = value;
        end
        
        function val = get.WeightL2Factor(this)
            val = this.PrivateLayer.Weights.L2Factor;
        end
        
        function this = set.WeightL2Factor(this, value)
            iValidateScalar(value);
            this.PrivateLayer.Weights.L2Factor = value;
        end
        
        function val = get.BiasL2Factor(this)
            val = this.PrivateLayer.Bias.L2Factor;
        end
        
        function this = set.BiasL2Factor(this, value)
            iValidateScalar(value);
            this.PrivateLayer.Bias.L2Factor = value;
        end
    end
    
    methods(Access = public)
        function this = Convolution2DLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
    end
    
    methods(Hidden, Static)
        function inputArguments = parseInputArguments(varargin)
            parser = iCreateParser();
            parser.parse(varargin{:});
            inputArguments = iConvertToCanonicalForm(parser);
        end
    end
    
    methods(Static)
        function this = loadobj(in)
            if in.Version <= 1
                in = iUpgradeVersionOneToVersionTwo(in);
            end
            this = iLoadConvolution2DLayerFromCurrentVersion(in);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            numFiltersString = int2str( sum(this.NumFilters) );
            filterSizeString = i2DSizeToString( this.FilterSize );
            if ~isequal(this.NumChannels, 'auto')
                % When using filter groups, the number of channels is
                % replicated to match NumFilters. For display, only show
                % the first element.
                numChannelsString = ['x' int2str( this.NumChannels(1) )];
            else
                numChannelsString = '';
            end
            strideString = int2str( this.Stride );
            paddingString = int2str( this.Padding );
            
            description = iGetMessageString( ...
                'nnet_cnn:layer:Convolution2DLayer:oneLineDisplay', ...
                numFiltersString, ...
                filterSizeString, ...
                numChannelsString, ...
                strideString, ...
                paddingString );
            
            type = iGetMessageString( 'nnet_cnn:layer:Convolution2DLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            hyperparameters = {
                'FilterSize'
                'NumChannels'
                'NumFilters'
                'Stride'
                'Padding'
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
        
        function tf = filterGroupsAreUsed(this)
            tf = numel(this.NumFilters) ~= 1;
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function sizeString = i2DSizeToString( sizeVector )
% i2DSizeToString   Convert a 2-D size stored in a vector of 2 elements
% into a string separated by 'x'.
sizeString = [ ...
    int2str( sizeVector(1) ) ...
    'x' ...
    int2str( sizeVector(2) ) ];
end

function p = iCreateParser()
p = inputParser;

defaultStride = 1;
defaultPadding = 0;
defaultNumChannels = 'auto';
defaultWeightLearnRateFactor = 1;
defaultBiasLearnRateFactor = 1;
defaultWeightL2Factor = 1;
defaultBiasL2Factor = 0;
defaultName = '';

p.addRequired('FilterSize', @iIsPositiveIntegerScalarOrRowVectorOfTwo);
p.addRequired('NumFilters', @iIsPositiveIntegerScalar);
p.addParameter('Stride', defaultStride, @iIsPositiveIntegerScalarOrRowVectorOfTwo);
p.addParameter('Padding', defaultPadding, @iIsNaturalNumberScalarOrRowVectorOfTwo);
p.addParameter('NumChannels', defaultNumChannels, @(x)iIsPositiveIntegerScalarOrThisString(x,'auto'));
p.addParameter('WeightLearnRateFactor', defaultWeightLearnRateFactor, @iIsFiniteRealNumericScalar);
p.addParameter('BiasLearnRateFactor', defaultBiasLearnRateFactor, @iIsFiniteRealNumericScalar);
p.addParameter('WeightL2Factor', defaultWeightL2Factor, @iIsFiniteRealNumericScalar);
p.addParameter('BiasL2Factor', defaultBiasL2Factor, @iIsFiniteRealNumericScalar);
p.addParameter('Name', defaultName, @iIsValidName);
end

function inputArguments = iConvertToCanonicalForm(p)
inputArguments = struct;
inputArguments.FilterSize = iMakeIntoRowVectorOfTwo(p.Results.FilterSize);
inputArguments.NumFilters = p.Results.NumFilters;
inputArguments.Stride = iMakeIntoRowVectorOfTwo(p.Results.Stride);
inputArguments.Padding = iMakeIntoRowVectorOfTwo(p.Results.Padding);
inputArguments.NumChannels = iConvertToEmptyIfAuto(p.Results.NumChannels);
inputArguments.WeightLearnRateFactor = p.Results.WeightLearnRateFactor;
inputArguments.BiasLearnRateFactor = p.Results.BiasLearnRateFactor;
inputArguments.WeightL2Factor = p.Results.WeightL2Factor;
inputArguments.BiasL2Factor = p.Results.BiasL2Factor;
inputArguments.Name = p.Results.Name;
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
    S.Weights.Value = gather(S.Weights.Value);
    S.Bias.Value = gather(S.Bias.Value);
catch e
    % Only throw the error we want to throw.
    e = MException( ...
        'nnet_cnn:layer:Convolution2DLayer:MustHaveGPUToLoadFrom2016a', ...
        getString(message('nnet_cnn:layer:Convolution2DLayer:MustHaveGPUToLoadFrom2016a')));
    throwAsCaller(e);
end
end

function obj = iLoadConvolution2DLayerFromCurrentVersion(in)
internalLayer = nnet.internal.cnn.layer.Convolution2D( ...
    in.Name, in.FilterSize, in.NumChannels, ...
    in.NumFilters, in.Stride, in.Padding);
internalLayer.Weights = iLoadLearnableParameter(in.Weights);
internalLayer.Bias = iLoadLearnableParameter(in.Bias);
            
obj = nnet.cnn.layer.Convolution2DLayer(internalLayer);
end

function learnableParameter = iLoadLearnableParameter(S)
% iLoadLearnableParameter   Load a learnable parameter from a structure S
learnableParameter = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
learnableParameter.Value = S.Value;
learnableParameter.LearnRateFactor = S.LearnRateFactor;
learnableParameter.L2Factor = S.L2Factor;
end

function tf = iIsPositiveIntegerScalarOrRowVectorOfTwo(x)
tf = all(x > 0) && iIsInteger(x) && (isscalar(x) || iIsRowVectorOfTwo(x));
end

function tf = iIsPositiveIntegerScalar(x)
tf = all(x > 0) && iIsInteger(x) && isscalar(x);
end

function tf = iIsNaturalNumberScalarOrRowVectorOfTwo(x)
tf = iIsNaturalNumber(x) && (isscalar(x) || iIsRowVectorOfTwo(x));
end

function tf = iIsPositiveIntegerScalarOrThisString(x, aString)
if(ischar(x))
    tf = strcmp(x,aString);
else
    tf = iIsPositiveIntegerScalar(x);
end
end

function tf = iIsFiniteRealNumericScalar(x)
tf = isscalar(x) && isfinite(x) && isreal(x) && isnumeric(x);
end

function tf = iIsValidName(x)
tf = ischar(x);
end


function tf = iIsInteger(x)
tf = isreal(x) && isnumeric(x) && all(mod(x,1)==0);
end

function tf = iIsNaturalNumber(x)
tf  = iIsInteger(x) && all(x >= 0);
end

function tf = iIsRowVectorOfTwo(x)
tf = isvector(x) && all(size(x) == [1 2]);
end

function iValidateScalar(value)
classes = {'numeric'};
attributes = {'scalar', 'nonempty'};
validateattributes(value, classes, attributes);
end

function rowVectorOfTwo = iMakeIntoRowVectorOfTwo(scalarOrRowVectorOfTwo)
if(iIsRowVectorOfTwo(scalarOrRowVectorOfTwo))
    rowVectorOfTwo = scalarOrRowVectorOfTwo;
else
    rowVectorOfTwo = [scalarOrRowVectorOfTwo scalarOrRowVectorOfTwo];
end
end

function y = iConvertToEmptyIfAuto(x)
if(iIsAutoString(x))
    y = [];
else
    y = x;
end
end

function tf = iIsAutoString(x)
tf = strcmp(x, 'auto');
end

function expectedNumChannels = iExpectedNumChannels(NumChannels)
expectedNumChannels = NumChannels;
if isequal(expectedNumChannels, 'auto')
    expectedNumChannels = NaN;
end
end