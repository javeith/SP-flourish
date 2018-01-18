classdef TrainingOptionsSGDM
    % TrainingOptionsSGDM   Training options for stochastic gradient descent with momentum
    %
    %   This class holds the training options for stochastic gradient 
    %   descent with momentum.
    %
    %   TrainingOptionsSGDM properties:
    %       Momentum                    - Momentum for learning.
    %       InitialLearnRate            - Initial learning rate.
    %       LearnRateScheduleSettings   - Settings for the learning rate
    %                                     schedule.
    %       L2Regularization            - Factor for L2 regularization.
    %       MaxEpochs                   - Maximum number of epochs.
    %       MiniBatchSize               - The size of a mini-batch for 
    %                                     training.
    %       Verbose                     - Flag for printing information to 
    %                                     the command window.
    %       CheckpointPath              - Path where checkpoint networks
    %                                     will be saved.
    %
    %   Example:
    %       Create a set of training options for training with stochastic
    %       gradient descent with momentum. The learning rate will be
    %       reduced by a factor of 0.2 every 5 epochs. The training will 
    %       last for 20 epochs, and each iteration will use a mini-batch 
    %       with 300 observations.
    %
    %       opts = trainingOptions('sgdm', ...
    %           'LearnRateSchedule', 'piecewise', ...
    %           'LearnRateDropFactor', 0.2, ... 
    %           'LearnRateDropPeriod', 5, ... 
    %           'MaxEpochs', 20, ... 
    %           'MiniBatchSize', 300);
    %
    %   See also trainingOptions, trainNetwork.
    
    % Copyright 2015-2016 The MathWorks, Inc.

    properties(SetAccess = private)
        % Momentum   Momentum for learning
        %   The momentum determines the contribution of the gradient step
        %   from the previous iteration to the current iteration of
        %   training. It must be a value between 0 and 1, where 0 will give
        %   no contribution from the previous step, and 1 will give a
        %   maximal contribution from the previous step.
        Momentum
        
        % InitialLearnRate   Initial learning rate
        %   The initial learning rate that is used for training. If the
        %   learning rate is too low, training will take a long time, but
        %   if it is too high, the training is likely to get stuck at a
        %   suboptimal result.
        InitialLearnRate
        
        % LearnRateScheduleSettings   Settings for the learning rate schedule
        %   The learning rate schedule settings. This summarizes the
        %   options for the chosen learning rate schedule. The field Method
        %   gives the name of the method for adjusting the learning rate.
        %   This can either be 'fixed', in which case the learning rate is
        %   not altered, or 'piecewise', in which case there will be two
        %   additional fields. These field are DropFactor, which is a 
        %   multiplicative factor for dropping the learning rate, and
        %   DropPeriod, which determines how many epochs should pass
        %   before dropping the learning rate.
        LearnRateScheduleSettings
        
        % L2Regularization   Factor for L2 regularization
        %   The factor for the L2 regularizer. It should be noted that each
        %   set of parameters in a layer can specify a multiplier for this
        %   L2 regularizer.
        L2Regularization
        
        % MaxEpochs   Maximum number of epochs
        %   The maximum number of epochs that will be used for training.
        %   Training will stop once this number of epochs has passed.
        MaxEpochs
        
        % MiniBatchSize   The size of a mini-batch for training
        %   The size of the mini-batch used for each training iteration.
        MiniBatchSize
        
        % Verbose   Flag for printing information to the command window
        %   If this is set to true, information on training progress will 
        %   be printed to the command window. The default is true.
        Verbose

        % Shuffle   This controls when the training data is shuffled. It
        % can either be 'once' to shuffle data once before training or
        % 'never' in order not to shuffle the data.
        Shuffle
        
        % CheckpointPath   This is the path where the checkpoint networks
        % will be saved. If empty, no checkpoint will be saved.
        CheckpointPath
    end
    
    methods(Access = public)
        function this = TrainingOptionsSGDM(inputArguments)
            this.Momentum = inputArguments.Momentum;
            this.InitialLearnRate = inputArguments.InitialLearnRate;
            this.LearnRateScheduleSettings = inputArguments.LearnRateScheduleSettings;
            this.L2Regularization = inputArguments.L2Regularization;
            this.MaxEpochs = inputArguments.MaxEpochs;
            this.MiniBatchSize = inputArguments.MiniBatchSize;
            this.Verbose = inputArguments.Verbose;
            this.Shuffle = inputArguments.Shuffle;
            this.CheckpointPath = inputArguments.CheckpointPath;
        end
    end
    
    methods(Static)
        function inputArguments = parseInputArguments(varargin)
            
            parser = iCreateParser();
            parser.parse(varargin{:});
            inputArguments = iPostProcessParsingResults(parser.Results);
        end
    end
end

function p = iCreateParser()

p = inputParser;

defaultMomentum = 0.9;
defaultInitialLearnRate = 0.01;
defaultLearnRateSchedule = 'none';
defaultLearnRateDropFactor = 0.1;
defaultLearnRateDropPeriod = 10;
defaultL2Regularization = 0.0001;
defaultMaxEpochs = 30;
defaultMiniBatchSize = 128;
defaultVerbose = true;
defaultShuffle = 'once';
defaultCheckpointPath = '';

p.addParameter('Momentum', defaultMomentum, @iIsRealNumericScalarBetweenZeroAndOneInclusive);
p.addParameter('InitialLearnRate', defaultInitialLearnRate, @iIsFinitePositiveFiniteRealNumericScalar);
p.addParameter('LearnRateSchedule', defaultLearnRateSchedule);
p.addParameter('LearnRateDropFactor', defaultLearnRateDropFactor, @iIsRealNumericScalarBetweenZeroAndOneInclusive);
p.addParameter('LearnRateDropPeriod', defaultLearnRateDropPeriod, @iIsPositiveIntegerScalar);
p.addParameter('L2Regularization', defaultL2Regularization, @iIsZeroOrFinitePositiveRealNumericScalar);
p.addParameter('MaxEpochs', defaultMaxEpochs, @iIsPositiveIntegerScalar);
p.addParameter('MiniBatchSize', defaultMiniBatchSize, @iIsPositiveIntegerScalar);
p.addParameter('Verbose', defaultVerbose, @iIsScalarAndLogicalOneOrZero);
p.addParameter('Shuffle', defaultShuffle);
p.addParameter('CheckpointPath', defaultCheckpointPath, @iIsValidCheckpointPath);

end

function inputArguments = iPostProcessParsingResults(results)
inputArguments = struct;
inputArguments.Momentum = results.Momentum;
inputArguments.InitialLearnRate = results.InitialLearnRate;
inputArguments.LearnRateScheduleSettings = iCreateLearnRateScheduleSettings( ...
    results.LearnRateSchedule, ...
    results.LearnRateDropFactor, ...
    results.LearnRateDropPeriod);
inputArguments.L2Regularization = results.L2Regularization;
inputArguments.MaxEpochs = results.MaxEpochs;
inputArguments.MiniBatchSize = results.MiniBatchSize;
inputArguments.Verbose = logical(results.Verbose);
inputArguments.Shuffle = iMatchWithValidShuffleValue(results.Shuffle);
inputArguments.CheckpointPath = results.CheckpointPath;
end

function scheduleSettings = iCreateLearnRateScheduleSettings( ...
    learnRateSchedule, learnRateDropFactor, learnRateDropPeriod)
scheduleSettings = struct;
learnRateSchedule = iMatchWithValidLearnRateSchedule(learnRateSchedule);
switch learnRateSchedule
    case 'none'
        scheduleSettings.Method = 'none';
    case 'piecewise'
        scheduleSettings.Method = 'piecewise';
        scheduleSettings.DropRateFactor = learnRateDropFactor;
        scheduleSettings.DropPeriod = learnRateDropPeriod;
    otherwise
        error(message('nnet_cnn:TrainingOptionsSGDM:InvalidLearningRateScheduleMethod'));
end
end

function tf = iIsValidCheckpointPath(x)
% iIsValidCheckpointPath   Return true if x is a valid checkpoint path or
% an empty string.
tf = ischar(x) && (isempty(x) || isdir(x));
end

function tf = iIsRealNumericScalarBetweenZeroAndOneInclusive(x)
tf = iIsRealNumericScalar(x) && (x >= 0) && (x <= 1);
end

function tf = iIsRealNumericScalar(x)
tf = isscalar(x) && isreal(x) && isnumeric(x);
end

function tf = iIsFinitePositiveFiniteRealNumericScalar(x)
tf = iIsFiniteRealNumericScalar(x) && (x > 0);
end

function tf = iIsFiniteRealNumericScalar(x)
tf = isfinite(x) && iIsRealNumericScalar(x);
end

function tf = iIsPositiveIntegerScalar(x)
tf = isscalar(x) && iIsInteger(x) && (x > 0);
end

function tf = iIsInteger(x)
tf = isreal(x) && isnumeric(x) && all(mod(x,1)==0);
end

function tf = iIsZeroOrFinitePositiveRealNumericScalar(x)
tf = iIsFiniteRealNumericScalar(x) && (x >= 0);
end

function tf = iIsScalarAndLogicalOneOrZero(x)
tf = isscalar(x) && iIsLogicalOneOrZero(x);
end

function tf =  iIsLogicalOneOrZero(x)
tf = islogical(x) || (x == 1) || (x == 0);
end

function shuffleValue = iMatchWithValidShuffleValue(x)
expectedShuffleValues = {'never', 'once'};
shuffleValue = validatestring(x, expectedShuffleValues, 'trainingOptions', 'Shuffle');
end

function learnRateScheduleValue = iMatchWithValidLearnRateSchedule(x)
expectedLearnRateScheduleValues = {'none', 'piecewise'};
learnRateScheduleValue = validatestring(x, expectedLearnRateScheduleValues, 'trainingOptions', 'LearnRateSchedule');
end