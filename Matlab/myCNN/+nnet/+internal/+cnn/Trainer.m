classdef Trainer
    % Trainer   Class for training a network
    
    %   Copyright 2015 The MathWorks, Inc.
    
    properties(Access = private)
        Options
        Schedule
        Precision
        Reporter
    end
    
    methods
        function this = Trainer(opts, precision, reporter)
            % Trainer    Constructor for a network trainer
            %
            % opts - training options (nnet.cnn.TrainingOptionsSGDM)
            % precision - data precision
            this.Options = opts;
            scheduleArguments = iGetArgumentsForScheduleCreation(opts.LearnRateScheduleSettings);
            this.Schedule = nnet.internal.cnn.LearnRateScheduleFactory.create(scheduleArguments{:});
            this.Precision = precision;
            this.Reporter = reporter;
        end
        
        function net = train(this, net, data)
            % train   Train a network
            %
            % Inputs
            %    net -- network to train
            %    data -- data encapsulated in a data dispatcher
            % Outputs
            %    net -- trained network
            reporter = this.Reporter;
            schedule = this.Schedule;
            maxEpochs = this.Options.MaxEpochs;
            
            momentum = this.Precision.cast( this.Options.Momentum );
            learnRate = this.Precision.cast( this.Options.InitialLearnRate );
            l2Regularization = this.Precision.cast( this.Options.L2Regularization );
            velocity = iInitializeVelocity(net, this.Precision);
            
            normalization = iGetNormalization(net);
            augmentations = iGetAugmentations(net);
            
            trainingTimer = tic;
            
            reporter.start();
            iteration = 0;
            this.shuffle( data );
            for epoch = 1:maxEpochs
                data.start();
                while ~data.IsDone
                    [X, Y] = data.next();
                    
                    X = apply([augmentations normalization], X);
                    
                    [gradients, miniBatchLoss, miniBatchAccuracy] = net.gradients(X, Y);
                    gradients = iNormalizeGradients(X, gradients);
                    
                    velocity = iCalculateVelocity( ...
                        momentum, velocity, ...
                        l2Regularization, net.LearnableParameters, ...
                        learnRate, gradients);
                    
                    net = net.updateLearnableParameters(velocity);
                    
                    elapsedTime = toc(trainingTimer);
                    
                    iteration = iteration + 1;
                    reporter.reportIteration( epoch, iteration, elapsedTime, miniBatchLoss, miniBatchAccuracy, learnRate );                   
                end
                learnRate = schedule.update(learnRate, epoch);
                
                reporter.reportEpoch( epoch, iteration, net );
            end
            reporter.finish();
        end
    end
    
    methods(Access = private)
        function shuffle(this,data)
            % shuffle   Shuffle the data at start of training as per
            % training options
            if isequal(this.Options.Shuffle, 'once')
                data.shuffle();
            end
        end
    end
end

function n = iGetNormalization(net)
if isempty(net.Layers)
    n = nnet.internal.cnn.layer.ImageTransform.empty;
else
    n = net.Layers{1}.Transforms;
end
end

function a = iGetAugmentations(net)
if isempty(net.Layers)
    a = nnet.internal.cnn.layer.ImageTransform.empty;
else
    a = net.Layers{1}.TrainTransforms;
end
end

function scheduleArguments = iGetArgumentsForScheduleCreation(learnRateScheduleSettings)
scheduleArguments = struct2cell(learnRateScheduleSettings);
end

function velocity = iInitializeVelocity(net, precision)
velocity = num2cell( precision.cast(zeros(numel(net.LearnableParameters),1)) );
end

function newVelocity = iCalculateVelocity(momentum, oldVelocity, globalL2Regularization, learnableParametersArray, globalLearnRate, gradients)

learnableParameters = iExtractLearnableParametersIntoCellArray(learnableParametersArray);
l2Factors = iExtractL2FactorsIntoCellArray(learnableParametersArray);
learnRateFactors = iExtractLearnRateFactorsIntoCellArray(learnableParametersArray);

numLearnableParameters = numel(learnableParametersArray);

newVelocity = cell(numLearnableParameters, 1);
for i = 1:numLearnableParameters
    newVelocity{i} = iVelocity(momentum,oldVelocity{i},globalL2Regularization,globalLearnRate,gradients{i},l2Factors{i},learnRateFactors{i},learnableParameters{i});
end

end

function vNew = iVelocity(m,vOld,gL2,gLR,grad,lL2,lLR,W)
% [1]   A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet
%       Classification with Deep Convolutional Neural Networks", in
%       Advances in Neural Information Processing Systems 25, 2012.

% m = momentum
% gL2 = globalL2Regularization
% gLR = globalLearnRate
% g = gradients, i.e., deriv of loss wrt weights
% lL2 = l2Factors, i.e., learn rate for a particular factor
% lLR = learnRateFactors
% W = learnableParameters

% learn rate for this parameters
alpha = gLR*lLR;
% L2 regularization for this parameters
lambda = gL2*lL2;

% Velocity formula as per [1]
vNew = m*vOld - lambda*alpha*W - alpha*grad;
end

function cellArray = iExtractLearnableParametersIntoCellArray(learnableParametersArray)
cellArray = cell(numel(learnableParametersArray),1);
for i = 1:numel(learnableParametersArray)
    cellArray{i} = learnableParametersArray(i).Value;
end
end

function cellArray = iExtractL2FactorsIntoCellArray(learnableParametersArray)
cellArray = cell(numel(learnableParametersArray),1);
for i = 1:numel(learnableParametersArray)
    cellArray{i} = learnableParametersArray(i).L2Factor;
end
end

function cellArray = iExtractLearnRateFactorsIntoCellArray(learnableParametersArray)
cellArray = cell(numel(learnableParametersArray),1);
for i = 1:numel(learnableParametersArray)
    cellArray{i} = learnableParametersArray(i).LearnRateFactor;
end
end

function gradients = iNormalizeGradients(X,gradients)
% Normalize the gradients by dividing by number of examples in the mini
% batch
n = size(X, 4);
gradients = cellfun(@(x)x/n, gradients, 'UniformOutput', false);
end
