classdef ReLULayer < nnet.cnn.layer.Layer
    % ReLULayer   Rectified linear unit (ReLU) layer
    %
    %   To create a rectified linear unit layer, use reluLayer
    %
    %   A rectified linear unit layer. This type of layer performs a simple
    %   thresholding operation, where any input value less than zero will
    %   be set to zero.
    %
    %   ReLULayer properties:
    %       Name                        - A name for the layer.
    %
    %   Example:
    %       Create a relu layer.
    %
    %       layer = reluLayer()
    %
    %   See also reluLayer
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties(SetAccess = private, Dependent)
        Name
    end
    
    methods
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function out = saveobj(this)
            out.Version = 1.0;
            out.Name = this.PrivateLayer.Name;
        end
    end
    
    methods(Static)
        function inputArguments = parseInputArguments(varargin)
            parser = iCreateParser();
            parser.parse(varargin{:});
            inputArguments = iConvertToCanonicalForm(parser);
        end
        
        function this = loadobj(in)
            internalLayer = nnet.internal.cnn.layer.ReLU(in.Name);
            this = nnet.cnn.layer.ReLULayer(internalLayer);
        end
    end
    
    methods(Access = public)
        function this = ReLULayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
    end
    
    methods(Access = protected)
        function [description, type] = getOneLineDisplay(~)
            description = iGetMessageString('nnet_cnn:layer:ReLULayer:oneLineDisplay');
            
            type = iGetMessageString( 'nnet_cnn:layer:ReLULayer:Type' );
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function p = iCreateParser()
p = inputParser;

defaultName = '';

addParameter(p, 'Name', defaultName, @iIsValidName);
end

function inputArguments = iConvertToCanonicalForm(p)
inputArguments = struct;
inputArguments.Name = p.Results.Name;
end

function tf = iIsValidName(x)
tf = ischar(x);
end