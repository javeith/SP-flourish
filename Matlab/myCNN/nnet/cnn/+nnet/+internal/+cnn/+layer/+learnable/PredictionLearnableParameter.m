classdef PredictionLearnableParameter < nnet.internal.cnn.layer.learnable.LearnableParameter
    % PredictionLearnableParameter   Learnable parameter for use at prediction time
    %
    %   This class is used to represent a learnable parameter at prediction
    %   time, and it is slightly more complex than the representation used
    %   at training time.
    %
    %   This class can be used in GPU mode or host mode, and this is
    %   controlled by setting the property "UseGPU".
    %
    %   1) When "UseGPU" is false, the "Value" property will return the 
    %   host array stored in "PrivateValue".
    %   2) When "UseGPU" is true, the "Value" property will return a
    %   gpuArray stored in "CacheHandle".
    
    %   Copyright 2016 The MathWorks, Inc
    
    properties(Dependent)
        % Value   The value of the learnable parameter
        Value
    end
    
    properties(Access = private)        
        % CacheHandle   A handle class which holds a copy of the learnable parameter on the GPU
        CacheHandle
    end
    
    properties(SetAccess = private)
        % HostValue   The value of the learnable parameter
        HostValue
    end
    
    properties
        % UseGPU   A boolean value which determines of the GPU is used
        UseGPU
        
        % LearnRateFactor   Multiplier for the learning rate for this parameter
        %   A scalar double.
        LearnRateFactor

        % L2Factor   Multiplier for the L2 regularizer for this parameter
        %   A scalar double.
        L2Factor
    end
    
    methods
        function this = PredictionLearnableParameter()
            this.UseGPU = false;
            this.CacheHandle = nnet.internal.cnn.layer.learnable.CacheHandle();
        end
        
        function this = set.Value(this, val)
            if(this.UseGPU)
                this = this.createNewCacheHandleIfOneExists();
            end
            this.HostValue = val;
        end
        
        function val = get.Value(this)
            if(this.UseGPU)
                this.populateCacheIfEmpty();
                val = this.CacheHandle.Value;
            else
                val = this.HostValue;
            end
        end
    end
    
    methods(Access = private)
        function this = createNewCacheHandleIfOneExists(this)
            % Create a new cache handle if one exists. Doing this ensures 
            % that any copies of the object get a new handle cache.
            if (~isempty(this.CacheHandle))
                this.CacheHandle = nnet.internal.cnn.layer.learnable.CacheHandle();
            end
        end
        
        function this = populateCacheIfEmpty(this)
            if (this.CacheHandle.isEmpty())
                this.CacheHandle.Value = gpuArray(this.HostValue);
            end
        end
    end
end
