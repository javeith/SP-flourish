classdef ProgressDisplayer < nnet.internal.cnn.util.Reporter
    properties(Constant)
        Frequency = 50; % iterations
    end
    
    methods
        function start( ~ )
            iPrintMessage('nnet_cnn:internal:cnn:Trainer:TableHorizontalBorder');
            iPrintMessage('nnet_cnn:internal:cnn:Trainer:TableHeadings');
            iPrintMessage('nnet_cnn:internal:cnn:Trainer:TableHorizontalBorder');
        end
        
        function reportIteration( this, epoch, iteration, elapsedTime, miniBatchLoss, miniBatchAccuracy, learnRate )
            if(mod(iteration, this.Frequency) == 0)
                fprintf('| %12d | %12d | %12.2f | %12.4f | %11.2f%% | %12f |\n', ...
                    epoch, iteration, elapsedTime, miniBatchLoss, miniBatchAccuracy, learnRate);
            end
        end
        
        function reportEpoch( ~, ~, ~, ~ )
        end
        
        function finish( ~ )
            iPrintMessage('nnet_cnn:internal:cnn:Trainer:TableHorizontalBorder');
        end
    end
    
end

function iPrintMessage(messageID, varargin)
string = getString(message(messageID, varargin{:}));
fprintf( '%s\n', string );
end
