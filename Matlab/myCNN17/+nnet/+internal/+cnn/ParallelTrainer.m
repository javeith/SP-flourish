classdef ParallelTrainer < nnet.internal.cnn.Trainer
    % ParallelTrainer   Class for training a network in parallel
    
    %   Copyright 2016-2017 The MathWorks, Inc.

    properties( Access = private )
        % InterruptStream  DataQueue created on the root worker that can be
        % used to send instructions to the pool to stop training
        InterruptStream
        
        % UseGpu  Whether to do optimized reductions on the GPU
        UseGpu
    end
    
    methods
        function this = ParallelTrainer(opts, precision, reporters, executionSettings)
            % ParallelTrainer    Constructor for a parallel trainer
            %
            % opts - training options (nnet.cnn.TrainingOptionsSGDM)
            % precision - data precision
            % reporters - reporters for feedback during training
            % executionSettings - training environment (eg host or GPU)
            
            % Wrap reporters with ParallelReporter which knows how to call
            % back to the client from the workers
            reporters = nnet.internal.cnn.util.ParallelReporter( reporters );
            
            % Construct superclass
            this@nnet.internal.cnn.Trainer(opts, precision, reporters, executionSettings);
            
            % Record execution environment
            this.UseGpu = ismember( executionSettings.executionEnvironment, {'gpu'} );
        end
        
        function net = train(this, net, data)
            % train   Train a network
            %
            % Inputs
            %    net -- network to train
            %    data -- data encapsulated in a data dispatcher
            % Outputs
            %    net -- trained network
            
            this.Reporter.start();
            
            % Record true endOfEpoch - on workers the dispatcher setting is
            % always 'truncateLast'. Also the overall minibatch size.
            dataSettings.endOfEpoch = data.EndOfEpoch;
            dataSettings.miniBatchSize = data.MiniBatchSize;
            
            % In order for SPMD to correctly understand how we wish to use
            % the Composite of data dispatchers stored within the
            % DistributedDataDispatcher, we must copy it out to a local
            % variable. We cannot access the containing dispatcher inside
            % SPMD.
            distributedData = data.DistributedData;
            
            % Determine which worker is going to have the output, and
            % create a DataQueue on that worker for the client to
            % communicate with the running trainer.
            [labIndexWithOutput, interruptStreamOnWorkers, this.InterruptStream] = ...
                iGetInterruptStream(distributedData);
            labIndexWithOutputOnClient = labIndexWithOutput;
            
            % The SPMD block executes the training algorithm independently
            % on each worker, with occasional collective communication to
            % merge network gradients and weights. Calling SPMD is costly,
            % so it must encompass the entire algorithm.
            spmd
                % This code executes on the workers

                % Create a communicator for just the enabled workers. This
                % prevents unnecessary communication with workers that
                % never process any data (because they are disabled).
                isWorkerActive = distributedData.NumObservations > 0;
                activeGroup = distributedutil.CommSplitter(double(isWorkerActive) + 1, labindex);
                
                % Now we can safely have disabled workers skip training
                if isWorkerActive
                    net = trainLocal(this, net, distributedData, ...
                                     dataSettings, interruptStreamOnWorkers);
                end
                
                % It is essential to destroy the communicator in order to
                % correctly synchronize the pool.
                delete(activeGroup);
                
                % Network returned should be ready for use on the client,
                % which may not have a GPU
                if labindex == labIndexWithOutput
                    net = net.setupNetworkForHostTraining();
                end
                
            end  % End of spmd block - now we are back on the client
            
            % Retrieve resulting network on the client
            net = net{labIndexWithOutputOnClient};
            
            this.Reporter.finish();
        end
                
        function net = finalizeNetwork(this, net, data)
            % Accumulate
            data = data.DistributedData;
            spmd
                % Make sure data is in right place
                if this.UseGpu
                    net = net.setupNetworkForGPUTraining(true);
                else
                    net = net.setupNetworkForHostTraining();
                end
                
                % Every worker needs to know which lab has the result
                hasData = data.NumObservations > 0;
                labIndexWithData = labindex;
                if ~hasData
                    labIndexWithData = inf;
                end
                labIndexWithResults = gop(@min, labIndexWithData);
                
                % Split the communicator so that only workers contributing data have to
                % communicate
                comm = feval('distributedutil.CommSplitter', double(hasData)+1, labindex);
                if hasData
                    % Call shared implementation to compute one final epoch
                    % on each worker - each worker will end up with a
                    % slightly different network.
                    net = this.doFinalize(net, data);

                    % Merge all finalizable layers, putting the result on
                    % the first worker in the split coomunicator.
                    for ii = 1:numel(net.Layers)
                        if isa(net.Layers{ii},'nnet.internal.cnn.layer.Finalizable')
                            net.Layers{ii} = gop( @mergeFinalized, net.Layers{ii}, 1 );
                        end
                    end
                end
                delete(comm);
                                
                % Network returned should be ready for use on the client,
                % which may not have a GPU
                if labindex == labIndexWithResults
                    net = net.setupNetworkForHostTraining();
                end

                % Copy the lab index with the result to the client using
                % AutoTransfer to avoid additional communication
                labIndexWithResults = feval('distributedutil.AutoTransfer', labIndexWithResults, labIndexWithResults );
            end
            % Gather back to the client
            net = net{labIndexWithResults.Value};
        end
    end
    
    methods( Access = protected )
        
        function stopTrainingCallback(this, ~, eventData)
        % stopTraining  Overload to send cancellation request to workers
            send(this.InterruptStream, eventData);
        end
    end
    
    methods( Access = private )
        
        function net = trainLocal(this, net, data, ds, interruptStream)
        % Training loop on a single worker
            trainingTimer = tic;
            prms = collectSettings(this, net);
            summary = nnet.internal.cnn.util.ParallelMiniBatchSummary;
            
            iteration = 0;
            this.StopTrainingFlag = false;
            [velocity, learnRate] = initializeLearning(this, net);
            
            needsStatefulTraining = false(numel(net.Layers), 1);
            for epoch = 1:prms.maxEpochs
                this.shuffle( data, prms.shuffleOption, epoch );
                data.start();
                while this.continueThisEpoch(data, interruptStream)
                    [X, response] = data.next();
                    
                    % Allow for finished or empty datastores
                    if ~isempty(X)
                        subBatchSize = size(X, 4);
                        % Cast data to appropriate execution environment for
                        % training
                        X = this.ExecutionStrategy.environment(X);
                        X = apply(prms.transforms, X);
                        propagateState = false;
                        [gradients, predictions] = this.computeGradients(net, X, response, needsStatefulTraining, propagateState);
                        miniBatchLoss = net.loss( predictions, response );
                    else
                        % For finished or empty datastores, use zero
                        % gradients. This ensures these workers receive
                        % gradients even though they are not contributing.
                        subBatchSize = 0;
                        gradients = iExtractZeroGradientsFromLearnableParameters( net.LearnableParameters );
                        predictions = [];
                        miniBatchLoss = 0;
                    end
                    % Compute overall minibatch size across pool
                    miniBatchSize = gplus(subBatchSize);
                    
                    % We may have used up all the data and not got
                    % enough for a whole minibatch. In this case if
                    % endOfEpoch param is 'discardLast', we skip.
                    if miniBatchSize < ds.miniBatchSize && ...
                            isequal( ds.endOfEpoch, 'discardLast' )
                        % This can only happen on all workers so all
                        % will exit the loop together
                        break;
                    end
                    
                    % Merge and normalize gradients between workers by
                    % summation
                    normalizationFactor = subBatchSize/miniBatchSize;
                    gradients = iMergeGradients( gradients, this.UseGpu, normalizationFactor );
                    
                    % Update weights - this happens entirely on the
                    % worker, but the results are the same on every
                    % worker thus resulting in the same networks.
                    velocity = calculateVelocity(this, ...
                        prms.momentum, velocity, ...
                        prms.l2Regularization, net.LearnableParameters, ...
                        learnRate, gradients);
                    net = net.updateLearnableParameters(velocity);
                    
                    % Update and report state
                    iteration = iteration + 1;
                    elapsedTime = toc(trainingTimer);
                    summary.update( predictions, response, ...
                        epoch, iteration, elapsedTime, ...
                        miniBatchLoss, learnRate, prms.lossFunctionType );
                    this.Reporter.computeIteration( summary, net );
                    this.Reporter.reportIteration( summary );
                end
                
                learnRate = this.Schedule.update(learnRate, epoch);
                
                this.Reporter.reportEpoch( epoch, iteration, net );
                
                % If an interrupt request has been made, break out of the
                % epoch loop
                if this.StopTrainingFlag
                    break;
                end
            end  % End of epoch loop
        end
        
        function continueEpoch = continueThisEpoch(this, data, interruptStream)
            % Tests when the epoch training loop should end based on the
            % data left in the datastores across the workers, and interrupt
            % requests. These two things are set together to keep the
            % communication to a minimum. Implemented as a method so that
            % the StopTrainingFlag property can be set, to cause the epoch
            % loop to be exited as well.
            [stopRequest, exception] = iReceivedInterrupt(interruptStream);
            stoppingConditions = ...
                struct('done', data.IsDone, 'stop', stopRequest, 'err', exception);
            stoppingConditions = gop(@iCatStruct, stoppingConditions);
            
            % If there were any errors, throw them collectively
            if ~isempty(stoppingConditions.err)
                throw( stoppingConditions.err(1) );
            end
            
            % We continue if any of the workers requested a stop or all
            % have finished their data
            isDone = all(stoppingConditions.done);
            this.StopTrainingFlag = any(stoppingConditions.stop);
            continueEpoch = ~isDone && ~this.StopTrainingFlag;
        end
        
    end
    
end

function [labIndexWithOutput, interruptStreamWorkers, interruptStreamClient] = ...
    iGetInterruptStream( distributedData )
% Determines which worker will hold the output network during training, and
% creates a DataQueue on that worker for the client to communicate with to
% control training.
spmd
    % The worker with the result will be the one with the
    % lowest rank that has any data
    isWorkerActive = distributedData.NumObservations > 0;
    labIndexWithOutput = labindex;
    if ~isWorkerActive
        labIndexWithOutput = inf;
    end
    labIndexWithOutput = gop(@min, labIndexWithOutput);
    
    % Create a DataQueue on this worker and return a Composite
    interruptStreamWorkers = parallel.internal.pool.DataQueue.empty;
    if labindex == labIndexWithOutput
        interruptStreamWorkers = parallel.internal.pool.DataQueue;
    end
    
    % Return results to client
    interruptStreamClient = distributedutil.AutoTransfer( interruptStreamWorkers, labIndexWithOutput );
    labIndexWithOutput = distributedutil.AutoTransfer( labIndexWithOutput, labIndexWithOutput );
end

% Retrieve underlying data
labIndexWithOutput = labIndexWithOutput.Value;
interruptStreamClient = interruptStreamClient.Value;
end

function [stop, exception] = iReceivedInterrupt( interruptStream )
% Checks the data queue on the root worker to see if an interrupt
% is being requested
stop = false;
exception = [];
if ~isempty(interruptStream)
    [data, ok] = poll(interruptStream);
    if ok
        stop = true;
        if isa( data, 'nnet.internal.cnn.util.TrainingInterruptEventData' )
            exception = data.Exception;
        end
    end
end
end

function zeroGradients = iExtractZeroGradientsFromLearnableParameters(learnableParametersArray)
zeroGradients = cell(numel(learnableParametersArray),1);
for i = 1:numel(learnableParametersArray)
    thisParam = learnableParametersArray(i).Value;
    zeroGradients{i} = zeros( size(thisParam), 'like', thisParam );
end
end

function gradients = iMergeGradients( gradients, useGpu, normalizationFactor )
% Adds gradients from all workers together, minimising communication and
% taking account of empty gradient arrays. Normalization relative to the
% true mini-batch size is handled by normalizationFactor, which is the
% quotient of the sub-batch size and the mini-batch size. Multiplication by
% this factor "undoes" normalization of gradients by the sub-batch size
% performed by the local loss layer.

% Concatenate all gradients so that we are communicating a single
% contiguous block of memory - this maximises the advantage of peer-to-peer
% communication for GPU devices
N = sum(cellfun(@(x)numel(x), gradients, 'UniformOutput', true));
localGradVec = zeros(N, 1, 'like', gradients{1});
i = 1;
for g = 1:numel(gradients)
    n = numel(gradients{g});
    localGradVec(i:(i+n-1)) = normalizationFactor.*gradients{g}(:);
    i = i + n;
end

% MPI all-reduce collective - all workers will have same gradients
if N > 0  % N will be the same on all workers
    % Use peer-to-peer optimization for GPU if possible
    if useGpu
        localGradVec = gplus(localGradVec, 'gpuArray');
    else
        localGradVec = gplus(localGradVec);
    end
end

% Put back into cell form
i = 1;
for g = 1:numel(gradients)
    N = numel(gradients{g});
    % Use reshape rather than subsasgn in case LHS is host and RHS is gpu
    gradients{g} = reshape(localGradVec(i:(i+N-1)), size(gradients{g}));
    i = i + N;
end
end

function s = iCatStruct(s1, s2)
% Concatenate the fields of two structures
s = s1;
f = fieldnames(s);
for i = 1:numel(f)
    fname = f{i};
    if ischar(s1.(fname))
        s.(fname) = {s1.(fname); s2.(fname)};
    else
        s.(fname) = [s1.(fname); s2.(fname)];
    end
end
end
