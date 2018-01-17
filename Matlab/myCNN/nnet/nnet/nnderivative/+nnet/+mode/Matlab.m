function mode = Matlab(varargin)
%MATLAB Matlab calculation mode

% Copyright 2012-2015 The MathWorks, Inc.

% Mode
mode.mode = 'nnet.mode.Matlab';

% Default Hints
hints.name = 'MATLAB';
hints.precision = 'data'; % 'data','double','single'
hints.direction = 'default'; % 'default','forward','backward'
hints.useGPU = 'data'; % 'data','no','yes','only'
hints.batchSize = Inf; % Inf, integer > 0

% Override Default Hints
mode.hints = nncalc.argPairs2Struct(hints,varargin);
mode.name = mode.hints.name;

% Info
mode.summary = @nnet.mode.matlab.summary;

% Mode setup
mode.netCheck = @nnet.mode.matlab.netCheck;
mode.netHints = @nnet.mode.matlab.netHints;
mode.dataHints = @nnet.mode.matlab.dataHints;
mode.codeHints = @nnet.mode.matlab.codeHints;
mode.formatData = @nnet.mode.matlab.formatData;
mode.formatNet = @nnet.mode.matlab.formatNet;

% Weights and biases
mode.setwb = @setwb;
mode.getwb = @getwb;

% Network evaluation
mode.pc = @nnet.mode.matlab.processInputs;
mode.y = @nnet.mode.matlab.calculateOutputs;

% Network performances
mode.trainPerf = @nnet.mode.matlab.calculatePerformance;
mode.trainValTestPerfs = @nnet.mode.matlab.calculatePerformance;

% Network gradients
mode.grad = @nnet.mode.matlab.calculateGradient;
mode.perfsGrad = @nnet.mode.matlab.calculateGradient;

% Network Jacobian
mode.perfsJEJJ = @nnet.mode.matlab.calculateJacobian;
