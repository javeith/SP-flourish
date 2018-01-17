classdef CrossChannelNormalizationLayer < nnet.cnn.layer.Layer
    % CrossChannelNormalizationLayer   Cross channel (local response) normalization layer
    %
    %   To create a cross channel normalization layer, use
    %   crossChannelNormalizationLayer. This type of layer is also known as
    %   local response normalization.
    %
    %   A cross channel normalization layer performs channel-wise
    %   normalization. For each element in the input x, we compute a
    %   normalized value y using the following formula:
    %
    %       y = x/(K + Alpha*ss/windowChannelSize)^Beta
    %
    %   where ss is the sum of squares of the elements in the normalization
    %   window. This function can be seen as a form of lateral inhibition
    %   between channels.
    %
    %   CrossChannelNormalizationLayer properties:
    %       Name                        - A name for the layer.
    %       WindowChannelSize            - The size of the channel window for
    %                                     normalization.
    %       Alpha                       - A multiplier for normalization
    %                                     term.
    %       Beta                        - The exponent for the normalzation
    %                                     term.
    %       K                           - An additive constant for the
    %                                     normalization term.
    %
    %   Example:
    %       Create a local response normalization layer for channel-wise
    %       normalization, where a window of 5 channels will be used to normalize
    %       each element, and the additive constant for the normalizer is 1.
    %
    %       layer = crossChannelNormalizationLayer(5, 'K', 1);
    %
    % [1]   A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet
    %       Classification with Deep Convolutional Neural Networks", in
    %       Advances in Neural Information Processing Systems 25, 2012.
    %
    %   See also crossChannelNormalizationLayer
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties(SetAccess = private, Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
        
        % WindowChannelSize   Size of the window for normalization
        %   The size of a window which controls the number of channels that are
        %   used for the normalization of each element. For example, if
        %   this value is 3, each element will be normalized by its
        %   neighbours in the previous channel and the next channel. If WindowChannelSize
        %   is even, then the window will be asymmetric. For example, if it
        %   is 4, each element is normalized by its neighbour in the
        %   previous channel, and by its neighbours in the next two channels.  The
        %   value must be a positive integer.
        WindowChannelSize
        
        % Alpha   Multiplier for the normalization term
        %   The Alpha term in the normalization formula.
        Alpha
        
        % Beta   Exponent for the normalization term
        %   The Beta term in the normalization formula.
        Beta
        
        % K   Additive constant for the normalization term
        %   The K term in the normalization formula.
        K
    end
    
    methods
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function val = get.WindowChannelSize(this)
            val = this.PrivateLayer.WindowChannelSize;
        end
        
        function val = get.Alpha(this)
            val = this.PrivateLayer.Alpha;
        end
        
        function val = get.Beta(this)
            val = this.PrivateLayer.Beta;
        end
        
        function val = get.K(this)
            val = this.PrivateLayer.K;
        end
    end
    
    methods(Hidden, Access = public)
        function this = CrossChannelNormalizationLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            out.Version = 1.0;
            out.Name = this.PrivateLayer.Name;
            out.WindowChannelSize = this.PrivateLayer.WindowChannelSize;
            out.Alpha = this.PrivateLayer.Alpha;
            out.Beta = this.PrivateLayer.Beta;
            out.K = this.PrivateLayer.K;
        end
    end
    
    methods(Hidden, Static)
        function inputArguments = parseInputArguments(varargin)
            parser = iCreateParser();
            parser.parse(varargin{:});
            inputArguments = iConvertToCanonicalForm(parser);
        end
        
        function this = loadobj(in)
            internalLayer = nnet.internal.cnn.layer.LocalMapNorm2D(in.Name, in.WindowChannelSize, in.Alpha, in.Beta, in.K);
            this = nnet.cnn.layer.CrossChannelNormalizationLayer(internalLayer);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            WindowChannelSizeString = int2str( this.WindowChannelSize );
            description = iGetMessageString( ...
                'nnet_cnn:layer:CrossChannelNormalizationLayer:oneLineDisplay', ...
                WindowChannelSizeString );
            
            type = iGetMessageString( 'nnet_cnn:layer:CrossChannelNormalizationLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            hyperparameters = {
                'WindowChannelSize'
                'Alpha'
                'Beta'
                'K'
                };
            
            groups = [
                this.propertyGroupGeneral( {'Name'} )
                this.propertyGroupHyperparameters( hyperparameters )
                ];
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function p = iCreateParser()
p = inputParser;

defaultAlpha = 0.0001;
defaultBeta = 0.75;
defaultK = 2;
defaultName = '';

[minWindowSize, maxWindowSize, minBeta, minK] = nnet.internal.cnngpu.localMapNormParamRanges();

p.addRequired('WindowChannelSize', iAssertValidWindowSizeFcn( minWindowSize, maxWindowSize ) )
p.addParameter('Alpha', defaultAlpha, @iAssertFiniteRealNumericScalar);
p.addParameter('Beta', defaultBeta, iAssertValidBetaFcn( minBeta ) );
p.addParameter('K', defaultK, iAssertValidKFcn( minK ) );
p.addParameter('Name', defaultName, @iIsValidName);
end

function fcn = iAssertValidWindowSizeFcn( minWindowSize, maxWindowSize )
% iAssertValidWindowSizeFcn   A function that will throw an error if given
% a window size that is not within the given range.
fcn = @nAssertValidWindowSize;

    function nAssertValidWindowSize( windowSize )
        iAssertIntegerScalar( windowSize );
        
        if windowSize<minWindowSize || windowSize>maxWindowSize
            error( message( ...
                'nnet_cnn:layer:CrossChannelNormalizationLayer:WrongWindowSize', ...
                mat2str([minWindowSize, maxWindowSize]) ) );
        end
    end
end

function fcn = iAssertValidBetaFcn( minBeta )
% iAssertValidBetaFcn   A function that will throw an error if given a
% value for beta that is not valid.
fcn = @nAssertValidBeta;

    function nAssertValidBeta( beta )
        iAssertFiniteRealNumericScalar( beta );
        
        if beta<minBeta
            error( message( ...
                'nnet_cnn:layer:CrossChannelNormalizationLayer:WrongBeta', ...
                mat2str(minBeta) ) );
        end
    end
end

function fcn = iAssertValidKFcn( minK )
% iAssertValidKFcn   A function that will throw an error if given a value
% for K that is not valid.
fcn = @nAssertValidK;

    function nAssertValidK( K )
        iAssertFiniteRealNumericScalar( K );
        
        if K<minK
            error( message( ...
                'nnet_cnn:layer:CrossChannelNormalizationLayer:WrongK', ...
                mat2str(minK) ) );
        end
    end
end

function inputArguments = iConvertToCanonicalForm(p)
inputArguments = struct;
inputArguments.WindowChannelSize = p.Results.WindowChannelSize;
inputArguments.Alpha = p.Results.Alpha;
inputArguments.Beta = p.Results.Beta;
inputArguments.K = p.Results.K;
inputArguments.Name = p.Results.Name;
end

function iAssertIntegerScalar(x)
if ~iIsIntegerScalar(x)
    error( message( ...
        'nnet_cnn:layer:CrossChannelNormalizationLayer:NotIntegerScalar' ) );
end
end

function iAssertFiniteRealNumericScalar(x)
if ~iIsFiniteRealNumericScalar(x)
    error( message( ...
        'nnet_cnn:layer:CrossChannelNormalizationLayer:NotFiniteRealNumericScalar' ) );
end
end

function tf = iIsInteger(x)
tf = isreal(x) && isnumeric(x) && all(mod(x,1)==0);
end

function tf = iIsIntegerScalar(x)
tf = isscalar(x) && iIsInteger(x);
end

function tf = iIsFiniteRealNumericScalar(x)
tf = isscalar(x) && isfinite(x) && isreal(x) && isnumeric(x);
end

function tf = iIsValidName(x)
tf = ischar(x);
end