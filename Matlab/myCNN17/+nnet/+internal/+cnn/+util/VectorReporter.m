classdef VectorReporter < nnet.internal.cnn.util.Reporter
    % VectorReporter   Container to hold a series of reporters
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties
        Reporters
    end
    
    methods
        function setup( this )
            for i = 1:length( this.Reporters )
                this.Reporters{i}.setup();
            end
        end
        
        function start( this )
            for i = 1:length( this.Reporters )
                this.Reporters{i}.start();
            end
        end
        
        function reportIteration( this, summary )
            for i = 1:length( this.Reporters )
                this.Reporters{i}.reportIteration( summary );
            end
        end
        
        function computeIteration( this, summary, network )
            for i = 1:length( this.Reporters )
                this.Reporters{i}.computeIteration( summary, network );
            end
        end
        
        function reportEpoch( this, epoch, iteration, network )
            for i = 1:length( this.Reporters )
                this.Reporters{i}.reportEpoch( epoch, iteration, network );
            end
        end
        
        function finish( this )
            for i = 1:length( this.Reporters )
                this.Reporters{i}.finish();
            end
        end
        
        function add( this, reporter )
            this.Reporters{end+1} = reporter;
            
            % A vector reporter must forward events fired by its members
            addlistener( reporter, 'TrainingInterruptEvent', ...
                @(~,~)notify(this, 'TrainingInterruptEvent') );
        end
    end
end
