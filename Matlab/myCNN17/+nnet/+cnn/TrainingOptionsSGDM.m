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
    %       VerboseFrequency            - This only has an effect if
    %                                     'Verbose' is set to true. It
    %                                     specifies the number of
    %                                     iterations between printing to
    %                                     the command window.
    %       ValidationData              - Data to use for validation during
    %                                     training.
    %       ValidationFrequency         - Number of iterations between
    %                                     evaluations of validation
    %                                     metrics.
    %       ValidationPatience          - The number of times that the
    %                                     validation loss is allowed to be
    %                                     larger than or equal to the
    %                                     previously smallest loss before
    %                                     training is stopped.
    %       Shuffle                     - This controls if the training
    %                                     data is shuffled.
    %       CheckpointPath              - Path where checkpoint networks
    %                                     will be saved.
    %       ExecutionEnvironment        - What hardware to use for training
    %                                     the network.
    %       WorkerLoad                  - Relative division of load between
    %                                     parallel workers on different
    %                                     hardware.
    %       OutputFcn                   - User callback to be executed at
    %                                     each iteration.
    %       Plots                       - Plots to display during training
    %       SequenceLength              - Sequence length of a mini-batch
    %                                     during training.
    %       SequencePaddingValue        - Value to pad mini-batches along
    %                                     the sequence dimension.
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
    
    % Copyright 2015-2017 The MathWorks, Inc.
    
    properties(Access = protected)
        % Version   Number to identify the current version of this object
        %   This is used to ensure that objects from older versions are
        %   loaded correctly.
        Version = 2
    end
    
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
        %   This can either be 'none', in which case the learning rate is
        %   not altered, or 'piecewise', in which case there will be two
        %   additional fields. These fields are DropFactor, which is a
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
        
        % VerboseFrequency   Frequency for printing information
        %   This only has an effect if 'Verbose' is true. It specifies the
        %   number of iterations between printing to the command window.
        VerboseFrequency
        
        % ValidationData   Data to be used for validation purposes
        ValidationData
        
        % ValidationFrequency   Frequency for computing validation metrics
        ValidationFrequency
        
        % ValidationPatience   Patience used to stop training
        ValidationPatience
        
        % Shuffle   This controls when the training data is shuffled. It
        % can either be 'once' to shuffle data once before training,
        % 'every-epoch' to shuffle before every training epoch, or 'never'
        % in order not to shuffle the data.
        Shuffle
        
        % CheckpointPath   This is the path where the checkpoint networks
        % will be saved. If empty, no checkpoint will be saved.
        CheckpointPath
        
        % ExecutionEnvironment   Determines what hardware to use for
        % training the network.
        ExecutionEnvironment
        
        % WorkerLoad   Relative division of load between parallel workers
        % on different hardware
        WorkerLoad
        
        % OutputFcn   Functions to call after each iteration, passing
        % training info from the current iteration, and returning true to
        % terminate training early
        OutputFcn
        
        % Plots   Plots to show during training
        Plots
        
        % SequenceLength   Determines the strategy used to create
        % mini-batches of sequence data
        SequenceLength
        
        % SequencePaddingValue   Scalar value used to pad mini-batches in
        % the along the sequence dimension
        SequencePaddingValue
        
    end
    
    methods(Access = public)
        function this = TrainingOptionsSGDM(inputArguments)
            this.Momentum = inputArguments.Momentum;
            this.InitialLearnRate = inputArguments.InitialLearnRate;
            this.LearnRateScheduleSettings = iCreateLearnRateScheduleSettings( ...
                inputArguments.LearnRateSchedule, ...
                inputArguments.LearnRateDropFactor, ...
                inputArguments.LearnRateDropPeriod);
            this.L2Regularization = inputArguments.L2Regularization;
            this.MaxEpochs = inputArguments.MaxEpochs;
            this.MiniBatchSize = inputArguments.MiniBatchSize;
            this.Verbose = inputArguments.Verbose;
            this.VerboseFrequency = inputArguments.VerboseFrequency;
            this.ValidationData = inputArguments.ValidationData;
            this.ValidationFrequency = inputArguments.ValidationFrequency;
            this.ValidationPatience = inputArguments.ValidationPatience;
            this.Shuffle = inputArguments.Shuffle;
            this.CheckpointPath = inputArguments.CheckpointPath;
            this.ExecutionEnvironment = inputArguments.ExecutionEnvironment;
            this.WorkerLoad = inputArguments.WorkerLoad;
            this.OutputFcn = inputArguments.OutputFcn;
            this.Plots = inputArguments.Plots;
            this.SequenceLength = inputArguments.SequenceLength;
            this.SequencePaddingValue = inputArguments.SequencePaddingValue;
        end
        
        function out = saveobj(this)
            out.Version = this.Version;
            out.Momentum = this.Momentum;
            out.InitialLearnRate = this.InitialLearnRate;
            out = iFlattenLearnRateScheduleSettings(out, this.LearnRateScheduleSettings);
            out.L2Regularization = this.L2Regularization;
            out.MaxEpochs = this.MaxEpochs;
            out.MiniBatchSize = this.MiniBatchSize;
            out.Verbose = this.Verbose;
            out.VerboseFrequency = this.VerboseFrequency;
            out.ValidationData = this.ValidationData;
            out.ValidationFrequency = this.ValidationFrequency;
            out.ValidationPatience = this.ValidationPatience;
            out.Shuffle = this.Shuffle;
            out.CheckpointPath = this.CheckpointPath;
            out.ExecutionEnvironment = this.ExecutionEnvironment;
            out.WorkerLoad = this.WorkerLoad;
            out.OutputFcn = this.OutputFcn;
            out.Plots = this.Plots;
            out.SequenceLength = this.SequenceLength;
            out.SequencePaddingValue = this.SequencePaddingValue;
        end
    end
    
    methods(Static)
        function inputArguments = parseInputArguments(varargin)
            try
                parser = iCreateParser();
                parser.parse(varargin{:});
                inputArguments = iConvertToCanonicalForm(parser);
            catch e
                % Reduce the stack trace of the error message by throwing as caller
                throwAsCaller(e)
            end
        end
        
        function this = loadobj(in)
            if iTrainingOptionsAreFrom2016aOr2016b(in)
                in = iUpgradeTrainingOptionsFrom2016aOr2016bTo2017a(in);
            end
            if iTrainingOptionsAreFrom2017a(in)
                in = iUpgradeTrainingOptionsFrom2017aTo2017b(in);
            end
            this = nnet.cnn.TrainingOptionsSGDM(in);
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
defaultVerboseFrequency = 50;
defaultValidationData = [];
defaultValidationFrequency = 50;
defaultValidationPatience = 5;
defaultShuffle = 'once';
defaultCheckpointPath = '';
defaultExecutionEnvironment = 'auto';
defaultWorkerLoad = [];
defaultOutputFcn = [];
defaultPlots = 'none';
defaultSequenceLength = 'longest';
defaultSequencePaddingValue = 0;

p.addParameter('Momentum', defaultMomentum, @iAssertValidMomentum);
p.addParameter('InitialLearnRate', defaultInitialLearnRate, @iAssertValidInitialLearnRate);
p.addParameter('LearnRateSchedule', defaultLearnRateSchedule, @(x)any(iAssertAndReturnValidLearnRateSchedule(x)));
p.addParameter('LearnRateDropFactor', defaultLearnRateDropFactor, @iAssertValidLearnRateDropFactor);
p.addParameter('LearnRateDropPeriod', defaultLearnRateDropPeriod, @iAssertIsPositiveIntegerScalar);
p.addParameter('L2Regularization', defaultL2Regularization, @iAssertValidL2Regularization);
p.addParameter('MaxEpochs', defaultMaxEpochs, @iAssertIsPositiveIntegerScalar);
p.addParameter('MiniBatchSize', defaultMiniBatchSize, @iAssertIsPositiveIntegerScalar);
p.addParameter('Verbose', defaultVerbose, @iAssertValidVerbose);
p.addParameter('VerboseFrequency', defaultVerboseFrequency, @iAssertIsPositiveIntegerScalar);
p.addParameter('ValidationData', defaultValidationData, @iAssertValidValidationData);
p.addParameter('ValidationFrequency', defaultValidationFrequency, @iAssertIsPositiveIntegerScalar);
p.addParameter('ValidationPatience', defaultValidationPatience, @iAssertValidValidationPatience);
p.addParameter('Shuffle', defaultShuffle, @(x)any(iAssertAndReturnValidShuffleValue(x)));
p.addParameter('CheckpointPath', defaultCheckpointPath, @iAssertValidCheckpointPath);
p.addParameter('ExecutionEnvironment', defaultExecutionEnvironment, @(x)any(iAssertAndReturnValidExecutionEnvironment(x)));
p.addParameter('WorkerLoad', defaultWorkerLoad, @iAssertValidWorkerLoad);
p.addParameter('OutputFcn', defaultOutputFcn, @iAssertValidOutputFcn);
p.addParameter('Plots', defaultPlots, @iAssertValidPlots);
p.addParameter('SequenceLength', defaultSequenceLength, @(x)any(iAssertAndReturnValidSequenceLength(x)) );
p.addParameter('SequencePaddingValue', defaultSequencePaddingValue, @iAssertValidSequencePaddingValue);
end

function inputArguments = iConvertToCanonicalForm(parser)
results = parser.Results;
inputArguments = struct;
inputArguments.Momentum = results.Momentum;
inputArguments.InitialLearnRate = results.InitialLearnRate;
inputArguments.LearnRateSchedule = results.LearnRateSchedule;
inputArguments.LearnRateDropFactor = results.LearnRateDropFactor;
inputArguments.LearnRateDropPeriod = results.LearnRateDropPeriod;
inputArguments.L2Regularization = results.L2Regularization;
inputArguments.MaxEpochs = results.MaxEpochs;
inputArguments.MiniBatchSize = results.MiniBatchSize;
inputArguments.Verbose = logical(results.Verbose);
inputArguments.VerboseFrequency = results.VerboseFrequency;
inputArguments.ValidationData = results.ValidationData;
inputArguments.ValidationFrequency = results.ValidationFrequency;
inputArguments.ValidationPatience = results.ValidationPatience;
inputArguments.Shuffle = iAssertAndReturnValidShuffleValue(results.Shuffle);
inputArguments.CheckpointPath = results.CheckpointPath;
inputArguments.ExecutionEnvironment = iAssertAndReturnValidExecutionEnvironment(results.ExecutionEnvironment);
inputArguments.WorkerLoad = results.WorkerLoad;
inputArguments.OutputFcn = results.OutputFcn;
inputArguments.Plots = iAssertAndReturnValidPlots(results.Plots);
inputArguments.SequenceLength = iAssertAndReturnValidSequenceLength(results.SequenceLength);
inputArguments.SequencePaddingValue = results.SequencePaddingValue;
end

function scheduleSettings = iCreateLearnRateScheduleSettings( ...
    learnRateSchedule, learnRateDropFactor, learnRateDropPeriod)
scheduleSettings = struct;
learnRateSchedule = iAssertAndReturnValidLearnRateSchedule(learnRateSchedule);
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

function out = iFlattenLearnRateScheduleSettings(out, learnRateScheduleSettings)
learnRateScedule = learnRateScheduleSettings.Method;
out.LearnRateSchedule = learnRateScedule;
switch learnRateScedule
    case 'none'
        out.LearnRateDropFactor = [];
        out.LearnRateDropPeriod = [];
    case 'piecewise'
        out.LearnRateDropFactor = learnRateScheduleSettings.DropRateFactor;
        out.LearnRateDropPeriod = learnRateScheduleSettings.DropPeriod;
    otherwise
        error(message('nnet_cnn:TrainingOptionsSGDM:InvalidLearningRateScheduleMethod'));
end
end

function tf = iTrainingOptionsAreFrom2016aOr2016b(in)
% For training options from 2016a and 2016b, "in" will be an object
% instead of a struct.
tf = isa(in, 'nnet.cnn.TrainingOptionsSGDM');
end

function tf = iTrainingOptionsAreFrom2017a(in)
% For training options from 2017a, Version will be 1
tf = in.Version == 1;
end

function inStruct = iUpgradeTrainingOptionsFrom2016aOr2016bTo2017a(in)
% iUpgradeTrainingOptionsFrom2016aOr2016bTo2017a   Upgrade training options
% from R2016a or R2016b to R2017a

% Set properties that exist in 2016a and 2016b
inStruct = struct;
inStruct.Momentum = in.Momentum;
inStruct.InitialLearnRate = in.InitialLearnRate;
inStruct = iFlattenLearnRateScheduleSettings(inStruct, in.LearnRateScheduleSettings);
inStruct.L2Regularization = in.L2Regularization;
inStruct.MaxEpochs = in.MaxEpochs;
inStruct.MiniBatchSize = in.MiniBatchSize;
inStruct.Verbose = in.Verbose;
inStruct.Shuffle = in.Shuffle;
inStruct.CheckpointPath = in.CheckpointPath;

% Set properties that don't exist in 2016a or 2016b
inStruct.VerboseFrequency = 50;
inStruct.ExecutionEnvironment = 'auto';
inStruct.WorkerLoad = [];
inStruct.OutputFcn = [];
inStruct.Version = 1;
end

function inStruct = iUpgradeTrainingOptionsFrom2017aTo2017b(inStruct)
% iUpgradeTrainingOptionsFrom2017aTo2017b   Upgrade training options
% from R2017a to R2017b

% Set properties that exist in 2017a
inStruct.Version = 2;

% Set properties that don't exist in 2017a
inStruct.ValidationData = [];
inStruct.ValidationFrequency = 50;
inStruct.ValidationPatience = 5;
inStruct.Plots = 'none';
inStruct.SequenceLength = 'longest';
inStruct.SequencePaddingValue = 0;
end

function iAssertValidCheckpointPath(x)
% iAssertValidCheckpointPath Throws an error if the checkpoint path is not
% valid. Valid checkpoint paths are empty strings and existing directories
% with write access.
isEmptyPath = isempty(x);
isWritableExistingDir = ischar(x) && isdir(x) && iCanWriteToDir(x);
isValidCheckpointPath = isEmptyPath || isWritableExistingDir;

if ~isValidCheckpointPath
    iThrowCheckpointPathError()
end
end

function iAssertValidMomentum(x)
validateattributes(x,{'numeric'}, ...
    {'scalar','real','finite','>=',0,'<=',1});
end

function iAssertValidInitialLearnRate(x)
validateattributes(x, {'numeric'}, ...
    {'scalar','real','finite','positive'});
end

function iAssertIsPositiveIntegerScalar(x)
validateattributes(x,{'numeric'}, ...
    {'scalar','integer','positive'});
end

function iAssertValidValidationPatience(x)
isValidPatience = isscalar(x) && (isinf(x) || isPositiveInteger(x));
if ~isValidPatience
    error(message('nnet_cnn:TrainingOptionsSGDM:InvalidValidationPatience'))
end
end

function tf = isPositiveInteger(x)
isPositive = x>0;
isInteger = isreal(x) && isnumeric(x) && all(mod(x,1)==0);
tf = isPositive && isInteger;
end

function iAssertValidL2Regularization(x)
validateattributes(x,{'numeric'}, ...
    {'scalar','real','finite','nonnegative'});
end

function iAssertValidVerbose(x)
validateattributes(x,{'logical','numeric'}, ...
    {'scalar','binary'});
end

function iAssertValidLearnRateDropFactor(x)
validateattributes(x,{'numeric'}, ...
    {'scalar','real','finite','>=',0,'<=',1});
end

function iAssertValidWorkerLoad(w)
if isempty(w)
    % an empty worker load value is valid
    return
end
validateattributes(w,{'numeric'}, ...
    {'vector','finite','nonnegative'});
if sum(w)<=0
    error(message('nnet_cnn:TrainingOptionsSGDM:InvalidWorkerLoad'));
end
end

function iAssertValidValidationData(validationData)
% iAssertValidValidationData   Return true if validationData is one of the
% allowed data types for validation. This can be either a table, an
% imageDatastore or a cell array containing two arrays. The consistency of
% the data with respect to training data and network architecture will be
% checked outside.
if istable(validationData)
    % data type is accepted, no further validation
elseif iIsAnImageDatastore(validationData)
    iAssertValidationDatastoreHasLabels(validationData);
elseif iscell(validationData)
    iAssertValidationCellDataHasTwoEntries(validationData)
else
    iThrowValidationDataError();
end
end

function tf = iIsAnImageDatastore(x)
tf = isa(x, 'matlab.io.datastore.ImageDatastore');
end

function iAssertValidationDatastoreHasLabels(imds)
if isempty(imds.Labels)
    error(message('nnet_cnn:TrainingOptionsSGDM:ImageDatastoreHasNoLabels'));
end
end

function iAssertValidationCellDataHasTwoEntries(dataCell)
if numel(dataCell)~=2
    error(message('nnet_cnn:TrainingOptionsSGDM:CellArrayNeedsTwoEntries'));
end
end

function iThrowValidationDataError()
error(message('nnet_cnn:TrainingOptionsSGDM:InvalidValidationDataType'));
end

function shuffleValue = iAssertAndReturnValidShuffleValue(x)
expectedShuffleValues = {'never', 'once', 'every-epoch'};
shuffleValue = validatestring(x, expectedShuffleValues);
end

function learnRateScheduleValue = iAssertAndReturnValidLearnRateSchedule(x)
expectedLearnRateScheduleValues = {'none', 'piecewise'};
learnRateScheduleValue = validatestring(x, expectedLearnRateScheduleValues);
end

function validString = iAssertAndReturnValidExecutionEnvironment(inputString)
validExecutionEnvironments = {'auto', 'gpu', 'cpu', 'multi-gpu', 'parallel'};
validString = validatestring(inputString, validExecutionEnvironments);
end

function iAssertValidPlots(inputString)
validPlots = {'training-progress', 'none'};
validatestring(inputString, validPlots);
end

function validString = iAssertAndReturnValidPlots(inputString)
validPlots = {'training-progress', 'none'};
validString = validatestring(inputString, validPlots, {'char', 'string'});
validString = char(validString);
end

function y = iAssertAndReturnValidSequenceLength( x )
try
    if ischar(x) || isstring(x)
        y = validatestring(x, {'longest', 'shortest'});
    else
        validateattributes(x, {'numeric'}, {'scalar', 'real', 'integer', 'positive'});
        y = x;
    end
catch
    error(message('nnet_cnn:TrainingOptionsSGDM:InvalidSequenceLength'));
end
end

function iAssertValidSequencePaddingValue( x )
validateattributes(x, {'numeric'}, {'scalar', 'real'} )
end

function iThrowCheckpointPathError()
error(message('nnet_cnn:TrainingOptionsSGDM:InvalidCheckpointPath'));
end

function tf = iCanWriteToDir(proposedDir)
[~, status] = fileattrib(proposedDir);
tf = status.UserWrite;
end

function iAssertValidOutputFcn(f)
isValidFcn = isempty(f) || iIsFunctionWithInputs(f) || iIsCellOfValidFunctions(f);
if ~isValidFcn
    error(message('nnet_cnn:TrainingOptionsSGDM:InvalidOutputFcn'));
end
end

function tf = iIsFunctionWithInputs( f )
tf = isa(f, 'function_handle') && nargin(f) ~= 0;
end

function tf = iIsCellOfValidFunctions(f)
tf = iscell(f) && all( cellfun(@iIsFunctionWithInputs,f(:)) );
end
