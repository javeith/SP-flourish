classdef VectorReporter < nnet.internal.cnn.util.Reporter
    properties
        Reporters
    end
    
    methods
        function start( this )
            for i = 1:length( this.Reporters )
                this.Reporters{i}.start();
            end
        end
        
        function reportIteration( this, epoch, iteration, elapsedTime, miniBatchLoss, miniBatchAccuracy, learnRate )
            for i = 1:length( this.Reporters )
                this.Reporters{i}.reportIteration( epoch, iteration, elapsedTime, miniBatchLoss, miniBatchAccuracy, learnRate );
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
        end
    end
end
