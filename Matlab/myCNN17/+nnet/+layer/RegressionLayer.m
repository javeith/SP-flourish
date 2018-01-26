classdef (Abstract) RegressionLayer < nnet.cnn.layer.Layer
    % RegressionLayer   Interface for regression layers
    %
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % Name (char vector)   A name for the layer
        Name = '';
        
        % ResponseNames (cellstr)   The names of the responses
        %   A cell array containing the names of the responses. This will
        %   be automatically determined at training time. Prior to
        %   training, it will be empty.
        ResponseNames
    end
    
    properties (SetAccess = protected)
        % Description (char vector)   A one line description for the layer
        Description
        
        % Type (char vector)   The type of layer
        Type
    end
    
    methods (Abstract)
        % forwardLoss    Return the loss between the output obtained from
        % the network and the expected output
        %
        % Inputs
        %   this - the output layer to forward the loss through
        %   Y - Predictions made by network
        %   T - Targets (actual values)
        %
        % Outputs
        %   loss - the loss between Y and T
        loss = forwardLoss( this, Y, T)
        
        % backwardLoss    Back propagate the derivative of the loss function
        %
        % Inputs
        %   this - the output layer to backprop the loss through
        %   Y - Predictions made by network
        %   T - Targets (actual values)
        %
        % Outputs
        %   dLdX - the derivative of the loss (L) with respect to the input X
        dLdX = backwardLoss( this, Y, T)
    end
    
    methods(Access = protected)
        function [description, type] = getOneLineDisplay( layer )
            if isempty( layer.Description )
                description = iGetMessageString( 'nnet_cnn:layer:RegressionLayer:oneLineDisplay' );
            else
                description = layer.Description;
            end
            
            if isempty( layer.Type )
                type = iGetMessageString( 'nnet_cnn:layer:RegressionLayer:Type' );
            else
                type = layer.Type;
            end
        end
    end
    
    methods
        function layer = set.Name( layer, val )
            iAssertValidLayerName( val );
            layer.Name = char( val );
        end
    end
end

function messageString = iGetMessageString( messageID )
messageString = getString( message( messageID ) );
end

function iAssertValidLayerName( name )
nnet.internal.cnn.layer.paramvalidation.validateLayerName( name );
end