classdef ClassificationOutputLayer < nnet.cnn.layer.Layer
    % ClassificationOutputLayer   Classification output layer
    %
    %   To create a classification output layer, use classificationLayer
    %
    %   A classification output layer. This layer is used as the output for
    %   a network that performs classification.
    %
    %   ClassificationOutputLayer properties:
    %       Name                        - A name for the layer.
    %       ClassNames                  - The names of the classes.
    %       OutputSize                  - The size of the output.
    %       LossFunction                - The loss function that is used
    %                                     for training.
    %
    %   Example:
    %       Create a classification output layer.
    %
    %       layer = classificationLayer();
    %
    %   See also classificationLayer
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties(SetAccess = private, Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
        
        % ClassNames   The names of the classes
        %   A cell array containing the names of the classes. This will be
        %   automatically determined at training time. Prior to training,
        %   it will be empty.
        ClassNames
        
        % OutputSize   The size of the output
        %   The size of the output. This will be determined at training
        %   time. Prior to training, it is set to 'auto'.
        OutputSize
        
        % LossFunction   The loss function for training
        %   The loss function that will be used during training. Possible
        %   values are:
        %       'crossentropyex'    - Cross-entropy for exclusive outputs.
        LossFunction
    end
    
    methods
        function val = get.OutputSize(this)
            if(isempty(this.PrivateLayer.NumClasses))
                val = 'auto';
            else
                val = this.PrivateLayer.NumClasses;
            end
        end
        
        function val = get.LossFunction(~)
            val = 'crossentropyex';
        end
        
        function val = get.ClassNames(this)
            val = this.PrivateLayer.ClassNames;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
    end
    
    methods(Access = public)
        function this = ClassificationOutputLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 1.0;
            out.Name = privateLayer.Name;
            numClasses = this.PrivateLayer.NumClasses;
            if isempty(numClasses)
                outputSize = [];
            else
                outputSize = [1 1 numClasses];
            end
            out.OutputSize = outputSize;
            out.ClassNames = privateLayer.ClassNames;
        end        
    end
    
    methods(Hidden, Static)
        function inputArguments = parseInputArguments(varargin)
            parser = iCreateParser();
            parser.parse(varargin{:});
            inputArguments = iConvertToCanonicalForm(parser);
        end
        
        function this = loadobj(in)
            if ~isempty(in.OutputSize)
                % Remove the first two singleton dimensions of the OutputSize to construct the internal layer.
                in.OutputSize = in.OutputSize(3);
            end
            internalLayer = nnet.internal.cnn.layer.CrossEntropy(in.Name, in.OutputSize);
            internalLayer.ClassNames = in.ClassNames;
            this = nnet.cnn.layer.ClassificationOutputLayer(internalLayer);
        end        
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            lossfunctionString = 'cross-entropy';
            
            numClasses = numel(this.ClassNames);
            
            if numClasses==0
                description = iGetMessageString( ...
                    'nnet_cnn:layer:ClassificationOutputLayer:oneLineDisplayNoClasses', ....
                    lossfunctionString );
            elseif numClasses==1
                description = iGetMessageString( ...
                    'nnet_cnn:layer:ClassificationOutputLayer:oneLineDisplayOneClass', ....
                    lossfunctionString, ...
                    this.ClassNames{1} );                
            elseif numClasses==2
                description = iGetMessageString( ...
                    'nnet_cnn:layer:ClassificationOutputLayer:oneLineDisplayTwoClasses', ....
                    lossfunctionString, ...
                    this.ClassNames{1}, ...
                    this.ClassNames{2} );                                
            elseif numClasses>=3
                description = iGetMessageString( ...
                    'nnet_cnn:layer:ClassificationOutputLayer:oneLineDisplayNClasses', ....
                    lossfunctionString, ...
                    this.ClassNames{1}, ...
                    this.ClassNames{2}, ...
                    int2str( numClasses-2 ) );                                
            end
            
            type = iGetMessageString( 'nnet_cnn:layer:ClassificationOutputLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            generalParameters = {
                'Name'
                'ClassNames'
                'OutputSize'
                };
            
            groups = [
                this.propertyGroupGeneral( generalParameters )
                this.propertyGroupHyperparameters( {'LossFunction'} )                
            ];
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
inputArguments.OutputSize = [];
inputArguments.Name = p.Results.Name;
end

function tf = iIsValidName(x)
tf = ischar(x);
end