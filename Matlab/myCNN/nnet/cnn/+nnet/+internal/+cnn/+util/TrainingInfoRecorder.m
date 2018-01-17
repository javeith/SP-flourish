classdef TrainingInfoRecorder < nnet.internal.cnn.util.Reporter
    properties(SetAccess = private)
        Info
    end
    
    methods
        function start( this )
            this.Info = iEmptyTrainingInfo();
        end
        
        function reportIteration( this, ~, iteration, ~, miniBatchLoss, miniBatchAccuracy, learnRate )
            this.Info = iSetTrainingInfo( this.Info, iteration, miniBatchLoss, miniBatchAccuracy, learnRate );
        end
        
        function reportEpoch( ~, ~, ~, ~ )
        end
        
        function finish( ~ )
        end
    end
end

function info = iEmptyTrainingInfo()
info = struct( ...
    'TrainingLoss', [], ...
    'TrainingAccuracy', [], ...
    'BaseLearnRate', []);
end

function info = iSetTrainingInfo(info, iteration, miniBatchLoss, miniBatchAccuracy, learnRate)
info.TrainingLoss(iteration) = gather(miniBatchLoss);
info.TrainingAccuracy(iteration) = gather(miniBatchAccuracy);
info.BaseLearnRate(iteration) = gather(learnRate);
end
