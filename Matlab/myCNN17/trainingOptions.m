function opts = trainingOptions(solverName, varargin)
% trainingOptions   Options for training a neural network
%
%   options = trainingOptions(solverName) creates a set of training options
%   for the solver specified by solverName. Possible values for solverName
%   include:
%
%       'sgdm'  -   Stochastic gradient descent with momentum.
%
%   options = trainingOptions(solverName, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the training
%   options:
%
%       'Momentum'            - This parameter only applies if the solver
%                               is 'sgdm'. The momentum determines the
%                               contribution of the gradient step from the
%                               previous iteration to the current
%                               iteration of training. It must be a value
%                               between 0 and 1, where 0 will give no
%                               contribution from the previous step, and 1
%                               will give a maximal contribution from the
%                               previous step. The default value is 0.9.
%       'InitialLearnRate'    - The initial learning rate that is used for
%                               training. If the learning rate is too low,
%                               training will take a long time, but if it
%                               is too high, the training is likely to get
%                               stuck at a suboptimal result. The default
%                               is 0.01.
%       'LearnRateSchedule'   - This option allows the user to specify a
%                               method for lowering the global learning
%                               rate during training. Possible options
%                               include:
%                                 - 'none' - The learning rate does not
%                                   change and remains constant.
%                                 - 'piecewise' - The learning rate is
%                                   multiplied by a factor every time a
%                                   certain number of epochs has passed.
%                                   The multiplicative factor is controlled
%                                   by the parameter 'LearnRateDropFactor',
%                                   and the number of epochs between
%                                   multiplications is controlled by
%                                   'LearnRateDropPeriod'.
%                               The default is 'none'.
%       'LearnRateDropFactor' - This parameter only applies if the
%                               'LearnRateSchedule' is set to 'piecewise'.
%                               It is a multiplicative factor that is
%                               applied to the learning rate every time a
%                               certain number of epochs has passed.
%                               The default is 0.1.
%       'LearnRateDropPeriod' - This parameter only applies if the
%                               'LearnRateSchedule' is set to 'piecewise'.
%                               The learning rate drop factor will be
%                               applied to the global learning rate every
%                               time this number of epochs is passed. The
%                               default is 10.
%       'L2Regularization'    - The factor for the L2 regularizer. It
%                               should be noted that each set of parameters
%                               in a layer can specify a multiplier for
%                               this L2 regularizer. The default is 0.0001.
%       'MaxEpochs'           - The maximum number of epochs that will be
%                               used for training. The default is 30.
%       'MiniBatchSize'       - The size of the mini-batch used for each
%                               training iteration. The default is 128.
%       'Verbose'             - If this is set to true, information on
%                               training progress will be printed to the
%                               command window. The default is true.
%       'VerboseFrequency'    - This only has an effect if 'Verbose' is set
%                               to true. It specifies the number of
%                               iterations between printing to the command
%                               window. The default is 50.
%       'ValidationData'      - Data to use for validation during training.
%                               This can be:
%                                 - An ImageDatastore with categorical
%                                   labels
%                                 - A table, where the first column 
%                                   contains either image paths or images
%                                 - A cell array {X, Y}, where X is a
%                                   numeric array with the input data and Y
%                                   is an array of responses
%       'ValidationFrequency' - Number of iterations between evaluations of
%                               validation metrics. This only has an effect
%                               if you also specify 'ValidationData'. The
%                               default is 50.
%       'ValidationPatience'  - Number of times that the validation loss is
%                               allowed to be larger than or equal to the
%                               previously smallest loss before network
%                               training is stopped, specified as a
%                               positive integer or Inf. The default is 5.
%                               To turn off automatic stopping of network
%                               training, specify Inf as the
%                               'ValidationPatience' value.
%       'Shuffle'             - This controls if the training data is
%                               shuffled. The options are:
%                                 - 'never'- No shuffling is applied.
%                                 - 'once' - The data will be shuffled once
%                                   before training.
%                                 - 'every-epoch' - The data will be
%                                   shuffled before every training epoch.
%                               The default is 'once'.
%       'CheckpointPath'      - The path where checkpoint networks are
%                               saved. When specified, the software saves
%                               checkpoint networks after every epoch.
%                               If not specified, no checkpoints are saved.
%       'ExecutionEnvironment'
%                             - The execution environment for the
%                               network. This determines what hardware
%                               resources will be used to train the
%                               network. To use GPUs or a compute cluster
%                               you must have Parallel Computing
%                               Toolbox(TM). GPUs must be CUDA-enabled with
%                               compute capability 3.0 or higher.
%                                 - 'auto' - Use a GPU if it is
%                                   available, otherwise use the CPU.
%                                 - 'gpu' - Use the GPU.
%                                 - 'cpu' - Use the CPU.
%                                 - 'multi-gpu' - Use multiple GPUs on one
%                                   machine, using a local parallel pool.
%                                   If no pool is open, one is opened with
%                                   one worker per supported GPU device.
%                                 - 'parallel' - Use a compute cluster. If
%                                   no pool is open, one is opened using
%                                   the default cluster profile. If the
%                                   pool has access to GPUs then they will
%                                   be used and excess workers will be
%                                   idle. If the pool has no GPUs then
%                                   training will take place on all cluster
%                                   CPUs.
%                               The default is 'auto'.
%       'WorkerLoad'          - For the 'multi-gpu' and 'parallel'
%                               execution environments. Determines how to
%                               divide up work between GPUs or CPUs.
%                               Expects a numeric vector with one value per
%                               worker in the parallel pool. For value W,
%                               each worker will get W/sum(W) of the work.
%                               The default is for work to be divided
%                               evenly. Use this advanced option to balance
%                               load between unevenly performing hardware.
%       'OutputFcn'           - Specifies one or more functions to be
%                               called during training at the end of each
%                               iteration. Typically, you might use an
%                               output function to display or plot progress
%                               information, or determine whether training
%                               should be terminated early. The function
%                               will be passed a struct containing
%                               information from the current iteration. It
%                               may also return true, which will trigger
%                               early termination.
%       'Plots'               - Plots to display during training, specified
%                               as 'training-progress' or 'none' (default)
%       'SequenceLength'      - Pad or truncate sequences in a mini-batch
%                               to a specified length. Options are:
%                                 - 'longest' - pad all sequences in a 
%                                   mini-batch to the length of the longest
%                                   sequence.
%                                 - 'shortest' - truncate all sequences in
%                                   a mini-batch to the length of the
%                                   shortest sequence.
%                                 - Positive integer - pad sequences to
%                                   the have same length as the longest
%                                   sequence, then split into smaller
%                                   sequences of the specified length. If
%                                   splitting occurs, then the function
%                                   creates extra mini-batches.
%                               The default is 'longest'.
%       'SequencePaddingValue' - Scalar value used to pad sequences where
%                                necessary. The default is 0.
%
%   Example:
%       Create a set of training options for training with stochastic
%       gradient descent with momentum. The learning rate will be reduced
%       by a factor of 0.2 every 5 epochs. The training will last for 20
%       epochs, and each iteration will use a mini-batch with 300
%       observations.
%
%       options = trainingOptions('sgdm', ...
%           'LearnRateSchedule', 'piecewise', ...
%           'LearnRateDropFactor', 0.2, ...
%           'LearnRateDropPeriod', 5, ...
%           'MaxEpochs', 20, ...
%           'MiniBatchSize', 300);
%
%   See also nnet.cnn.TrainingOptionsSGDM, trainNetwork.

%   Copyright 2015-2017 The MathWorks, Inc.

if(strcmp(solverName,'sgdm'))
    args = nnet.cnn.TrainingOptionsSGDM.parseInputArguments(varargin{:});
    opts = nnet.cnn.TrainingOptionsSGDM(args);
else
    exception = iCreateExceptionFromErrorID('nnet_cnn:trainingOptions:InvalidSolverName');
    throwAsCaller(exception);
end
end

function exception = iCreateExceptionFromErrorID(errorID)
exception = MException(errorID, getString(message(errorID)));
end
