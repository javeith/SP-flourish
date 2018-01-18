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
%       'Shuffle'             - This controls if the training data is
%                               shuffled. The options are:
%                                 - 'once' - The data will be shuffled once
%                                   before training.
%                                 - 'never'- No shuffling is applied.
%                               The default is 'once'.
%       'CheckpointPath'      - The path where checkpoint networks are
%                               saved. When specified, the software saves 
%                               checkpoint networks after every epoch.
%                               If not specified, no checkpoints are saved.
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

%   Copyright 2015-2016 The MathWorks, Inc.

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