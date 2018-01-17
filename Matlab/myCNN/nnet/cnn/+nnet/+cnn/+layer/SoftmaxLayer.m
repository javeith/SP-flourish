classdef SoftmaxLayer < nnet.cnn.layer.Layer
    % SoftmaxLayer   Softmax layer
    %
    %   A softmax layer. This layer is useful for classification problems.
    %
    %   SoftmaxLayer properties:
    %       Name                    - A name for the layer
    %
    %   Example:
    %       Create a softmax layer.
    %
    %       layer = softmaxLayer();
    %
    %   See also softmaxLayer
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties(SetAccess = private, Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end
    
    methods
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
    end
    
    methods(Access = public)
        function this = SoftmaxLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            out.Version = 1.0;
            out.Name = this.PrivateLayer.Name;
        end        
    end
    
    methods(Hidden, Static)
        function inputArguments = parseInputArguments(varargin) 
            parser = iCreateParser();
            parser.parse(varargin{:});
            inputArguments = iConvertToCanonicalForm(parser);
        end
        
        function this = loadobj(in)
            internalLayer = nnet.internal.cnn.layer.Softmax(in.Name);
            this = nnet.cnn.layer.SoftmaxLayer(internalLayer);
        end        
    end 
    
    methods(Access = protected)
        function [description, type] = getOneLineDisplay(~)
            description = iGetMessageString('nnet_cnn:layer:SoftmaxLayer:oneLineDisplay');
            
            type = iGetMessageString( 'nnet_cnn:layer:SoftmaxLayer:Type' );
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