classdef (Sealed) ValidationReporter < nnet.internal.cnn.util.Reporter
    % ValidationReporter   Class to hold validation data and compute
    % performance metrics on them
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Access = private)
        % Data (nnet.internal.cnn.DataDispatcher)   A data dispatcher
        Data
        
        % Precision (nnet.internal.cnn.util.Precision)   A precision object
        Precision
        
        % ExecutionEnvironment   A char vector specifying the execution
        % environment
        ExecutionEnvironment
        
        % Frequency   Frequency to compute validation metrics in iterations
        Frequency
        
        % Patience   Number of times that the loss is allowed to increase
        % or remain unchanged before training is stopped
        Patience
        
        % BestLoss   Smallest loss achieved by the network so far
        BestLoss = Inf;
        
        % StepsWithoutDecrease   Number of steps passed without a decrease
        % of the best loss
        StepsWithoutDecrease = 0;
        
        % ShuffleOption   Controls when to shuffle the data. It can be:
        % 'once', 'every-epoch', or 'never'
        ShuffleOption
    end
    
    methods
        function this = ValidationReporter(data, precision, executionEnvironment, frequency, patience, shuffleOption)
            this.Data = data;
            this.Precision = precision;
            this.ExecutionEnvironment = executionEnvironment;
            this.Frequency = frequency;
            this.Patience = patience;
            this.ShuffleOption = shuffleOption;
            
            if isequal(this.ShuffleOption, 'once')
                this.Data.shuffle();                
            end
        end
        
        function setup( ~ )
        end
        
        function start( ~ )
        end
        
        function reportIteration( this, summary )
            if this.canCompute( summary.Iteration )
                this.updateBestLoss( summary.ValidationLoss );
                this.notifyTrainingInterrupt();
            end
        end
        
        function computeIteration(this, summary, net)
            % computeIteration   Compute predictions on the validation set
            % using the network net according to the current iteration and
            % update the MiniBatchSummary summary
            %
            % summary   - A nnet.internal.cnn.util.MiniBatchSummary object
            % net       - A nnet.internal.cnn.SeriesNetwork object
            
            if this.canCompute( summary.Iteration )
                [predictions, response] = this.predict( net );
                loss = net.loss( predictions, response );
            else
                % Return empty values
                predictions = [];
                response = [];
                loss = [];
            end
            summary.ValidationPredictions = predictions;
            summary.ValidationResponse = response;
            summary.ValidationLoss = loss;
        end
        
        function reportEpoch( ~, ~, ~, ~ )
        end
        
        function finish( ~ )
        end
        
        function computeFinalValidationResultForPlot(this, summary, net)
            % This method should only ever be called after network has
            % finished training, and has been setup for host prediction.
            summary.Iteration = 1;
            this.ExecutionEnvironment = 'cpu';
            this.computeIteration(summary, net);
        end
    end
    
    methods (Access = private)
        function tf = canCompute( this, iteration )
            tf = mod(iteration, this.Frequency) == 0 || iteration == 1;
        end
        
        function [predictions, response] = predict( this, net )
            predictions = this.allocatePredictionsArray();
            response = this.allocatePredictionsArray();
            % When 'shuffle' is set to 'every-epoch', validation data is
            % shuffled everytime we compute validation metrics. This is
            % because one epoch corresponds to an entire pass over the
            % dataset, so each time we compute a validation iteration using
            % all the data we should shuffle
            if isequal(this.ShuffleOption, 'every-epoch')
                this.Data.shuffle();
            end
            this.Data.start();
            while ~this.Data.IsDone
                [X, Y, idx] = this.Data.next();
                X = this.prepareForExecutionEnvironment(X);
                Y = this.prepareForExecutionEnvironment(Y);
                currentBatchPredictions = net.predict( X );
                % Predictions can be in a cell-array if net is a DAG
                % network
                if iscell( currentBatchPredictions )
                    currentBatchPredictions = currentBatchPredictions{1};
                end
                predictions(:,:,:,idx) = currentBatchPredictions;
                response(:,:,:,idx) = Y;
            end
        end
        
        function X = prepareForExecutionEnvironment( this, X )
            if nnet.internal.cnn.util.GPUShouldBeUsed( this.ExecutionEnvironment )
                % Move data to GPU
                X = gpuArray( X );
            else
                % Do nothing
            end
        end
        
        function predictions = allocatePredictionsArray( this )
            % allocatePredictionsArray   Allocate a prediction array
            % according to the size of the responses and the number of
            % observations
            predictions = this.Precision.zeros([this.Data.ResponseSize this.Data.NumObservations]);
            predictions = this.prepareForExecutionEnvironment( predictions );
        end
        
        function updateBestLoss( this, loss )
            % updateBestLoss   If the loss has decreased, update the best
            % loss. If it hasn't, increase the counter of steps without
            % loss decrease
            
            if loss < this.BestLoss
                this.BestLoss = loss;
                this.StepsWithoutDecrease = 0;
            else
                this.StepsWithoutDecrease = this.StepsWithoutDecrease + 1;
            end
        end
        
        function notifyTrainingInterrupt( this )
            % notifyTrainingInterrupt   Notify listeners of
            % TrainingInterruptEvent if there was no loss decrease for
            % this.Patience steps
            if this.StepsWithoutDecrease >= this.Patience
                notify( this, 'TrainingInterruptEvent' );
            end
        end
    end
end

