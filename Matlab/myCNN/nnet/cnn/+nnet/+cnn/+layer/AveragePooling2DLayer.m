classdef AveragePooling2DLayer < nnet.cnn.layer.Layer
    % AveragePooling2DLayer   Average pooling layer
    %
    %   To create a 2d average pooling layer, use averagePooling2dLayer
    %
    %   An average pooling layer. This layer performs downsampling.
    %
    %   AveragePooling2DLayer properties:
    %       Name                    - A name for the layer.
    %       PoolSize                - The height and width of pooling
    %                                 regions.
    %       Stride                  - The vertical and horizontal stride.
    %       Padding                 - The vertical and horizontal padding.
    %
    %   Example:
    %       Create an average pooling layer with non-overlapping pooling
    %       regions, which downsamples by a factor of 2:
    %
    %       layer = averagePooling2dLayer(2, 'Stride', 2);
    %
    %   See also averagePooling2dLayer

    
    %   Copyright 2015-2016 The MathWorks, Inc.

    properties(SetAccess = private, Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name

        % PoolSize   The height and width of a pooling region
        %   The size the pooling regions. This is a vector [h w] where h is
        %   the height of a pooling region, and w is the width of a pooling
        %   region.
        PoolSize
        
        % Stride   The vertical and horizontal stride
        %   The step size for traversing the input vertically and
        %   horizontally. This is a vector [u v] where u is the vertical
        %   stride and v is the horizontal stride.
        Stride
        
        % Padding   The vertical and horizontal padding
        %   The padding that is applied to the input for this layer. This
        %   is a vector [a b] where a is the padding applied to the top and
        %   bottom of the input, and b is the padding applied to the left
        %   and right.
        Padding
    end
    
    methods
        function val = get.PoolSize(this)
            val = this.PrivateLayer.PoolSize;
        end
        
        function val = get.Stride(this)
            val = this.PrivateLayer.Stride;
        end
        
        function val = get.Padding(this)
            val = this.PrivateLayer.Padding;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
    end
    
    methods(Access = public)
        function this = AveragePooling2DLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;            
            out.Version = 1.0;
            out.Name = privateLayer.Name;
            out.PoolSize = privateLayer.PoolSize;
            out.Stride = privateLayer.Stride;
            out.Padding = privateLayer.Padding;
        end        
    end
    
    methods(Hidden, Static)
        function inputArguments = parseInputArguments(varargin)        
            parser = iCreateParser();
            parser.parse(varargin{:});
            inputArguments = iConvertToCanonicalForm(parser);
            iAssertPoolSizeIsGreaterThanPaddingSize(inputArguments);
        end
        
        function this = loadobj(in)
            internalLayer = nnet.internal.cnn.layer.AveragePooling2D(...
                in.Name, in.PoolSize, in.Stride, in.Padding);
            this = nnet.cnn.layer.AveragePooling2DLayer(internalLayer);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            poolSizeString = i2DSizeToString( this.PoolSize );
            strideString = int2str( this.Stride );
            paddingString = int2str( this.Padding );
            
            description = iGetMessageString(  ...
                'nnet_cnn:layer:AveragePooling2DLayer:oneLineDisplay', ...
                poolSizeString, ...
                strideString, ...
                paddingString);
            
            type = iGetMessageString( 'nnet_cnn:layer:AveragePooling2DLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            hyperparameters = {
                'PoolSize'
                'Stride'
                'Padding'
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
defaultName = '';

addRequired(p, 'PoolSize', @iIsPositiveIntegerScalarOrRowVectorOfTwo);
addParameter(p, 'Stride', defaultStride, @iIsPositiveIntegerScalarOrRowVectorOfTwo);
addParameter(p, 'Padding', defaultPadding, @iIsNaturalNumberScalarOrRowVectorOfTwo);
addParameter(p, 'Name', defaultName, @iIsValidName);
end

function inputArguments = iConvertToCanonicalForm(p)
inputArguments = struct;
inputArguments.PoolSize = iMakeIntoRowVectorOfTwo(p.Results.PoolSize);
inputArguments.Stride = iMakeIntoRowVectorOfTwo(p.Results.Stride);
inputArguments.Padding = iMakeIntoRowVectorOfTwo(p.Results.Padding);
inputArguments.Name = p.Results.Name;
end

function tf = iIsPositiveIntegerScalarOrRowVectorOfTwo(x)
tf = all(x > 0) && iIsInteger(x) && (isscalar(x) || iIsRowVectorOfTwo(x));
end

function tf = iIsNaturalNumberScalarOrRowVectorOfTwo(x)
tf = iIsNaturalNumber(x) && (isscalar(x) || iIsRowVectorOfTwo(x));
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

function rowVectorOfTwo = iMakeIntoRowVectorOfTwo(scalarOrRowVectorOfTwo)
if(iIsRowVectorOfTwo(scalarOrRowVectorOfTwo))
    rowVectorOfTwo = scalarOrRowVectorOfTwo;
else
    rowVectorOfTwo = [scalarOrRowVectorOfTwo scalarOrRowVectorOfTwo];
end
end

function iAssertPoolSizeIsGreaterThanPaddingSize(inputArguments)
if(~iPoolSizeIsGreaterThanPaddingSize(inputArguments.PoolSize, inputArguments.Padding))
    error(message('nnet_cnn:layer:AveragePooling2DLayer:PaddingSizeLargerThanOrEqualToPoolSize'));
end
end

function tf = iPoolSizeIsGreaterThanPaddingSize(poolSize, padding)
tf = all(poolSize > padding);
end