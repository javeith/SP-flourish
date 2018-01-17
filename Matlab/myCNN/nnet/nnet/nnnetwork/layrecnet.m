function out1 = layrecnet(varargin)
%LAYRECNET Layered recurrent neural network.
%
%  Layer recurrent networks with two (or more) layers can learn to
%  predict any dynamic output from past inputs given enough hidden
%  neurons and enough recurrent layer delays.
%
%  <a href="matlab:doc layrecnet">layrecnet</a>(layerDelays,hiddenSizes,trainFcn) takes a row vectors
%  of layers delays, a row vector of hidden layer sizes, and a
%  backpropagation training function, and returns a layer recurrent neural
%  network with N+1 layers.
%
%  Input, output and output layers sizes are set to 0.  These sizes will
%  automatically be configured to match particular data by <a href="matlab:doc train">train</a>. Or the
%  user can manually configure inputs and outputs with <a href="matlab:doc configure">configure</a>.
%
%  Defaults are used if <a href="matlab:doc layrecnet">layrecnet</a> is called with fewer arguments.
%  The default arguments are (1:2,10,'<a href="matlab:doc trainlm">trainlm</a>').
%
%  Here a layer recurrent network is used to solve a time series problem.
%
%    [X,T] = <a href="matlab:doc simpleseries_dataset">simpleseries_dataset</a>;
%    net = <a href="matlab:doc layrecnet">layrecnet</a>(1:2,10);
%    [Xs,Xi,Ai,Ts] = <a href="matlab:doc preparets">preparets</a>(net,X,T);
%    net = <a href="matlab:doc train">train</a>(net,Xs,Ts,Xi,Ai);
%    <a href="matlab:doc view">view</a>(net);
%    Y = net(Xs,Xi,Ai);
%    perf = <a href="matlab:doc perform">perform</a>(net,Y,Ts)
%
%  To predict the next output a step ahead of when it will occur:
%
%    net = <a href="matlab:doc removedelay">removedelay</a>(net);
%    [Xs,Xi,Ai,Ts] = <a href="matlab:doc preparets">preparets</a>(net,X,T);
%    Y = net(Xs,Xi,Ai);
%
%  See also NARXNET, TIMEDELAYNET, DISTDELAYNET.

% Mark Beale, 2008-13-2008
% Copyright 2008-2011 The MathWorks, Inc.

%% =======================================================
%  BOILERPLATE_START
%  This code is the same for all Network Functions.

  persistent INFO;
  if isempty(INFO), INFO = get_info; end
  if (nargin > 0) && ischar(varargin{1}) ...
      && ~strcmpi(varargin{1},'hardlim') && ~strcmpi(varargin{1},'hardlims')
    code = varargin{1};
    switch code
      case 'info',
        out1 = INFO;
      case 'check_param'
        err = check_param(varargin{2});
        if ~isempty(err), nnerr.throw('Args',err); end
        out1 = err;
      case 'create'
        if nargin < 2, error(message('nnet:Args:NotEnough')); end
        param = varargin{2};
        err = nntest.param(INFO.parameters,param);
        if ~isempty(err), nnerr.throw('Args',err); end
        out1 = create_network(param);
        out1.name = INFO.name;
      otherwise,
        % Quick info field access
        try
          out1 = eval(['INFO.' code]);
        catch %#ok<CTCH>
          nnerr.throw(['Unrecognized argument: ''' code ''''])
        end
    end
  else
    [args,param] = nnparam.extract_param(varargin,INFO.defaultParam);
    [param,err] = INFO.overrideStructure(param,args);
    if ~isempty(err), nnerr.throw('Args',err,'Parameters'); end
    net = create_network(param);
    net.name = INFO.name;
    out1 = init(net);
  end
end

function v = fcnversion
  v = 7;
end

%  BOILERPLATE_END
%% =======================================================

function info = get_info
  info = nnfcnNetwork(mfilename,'Layer Recurrent Neural Network',fcnversion, ...
    [ ...
    nnetParamInfo('layerDelays','Layer Delays','nntype.strictpos_delayvec',1:2,...
    'Row vector delays in each layers feedback connection.'), ...
    nnetParamInfo('hiddenSizes','Hidden Layer Sizes','nntype.strict_pos_int_row',10,...
    'Sizes of 0 or more hidden layers.'), ...
    nnetParamInfo('trainFcn','Training Function','nntype.training_fcn','trainlm',...
    'Function to train the network.'), ...
    ]);
end

function err = check_param(param)
  err = '';
end

function net = create_network(param)
  net = feedforwardnet(param.hiddenSizes,param.trainFcn);
  for i=1:(net.numLayers-1)
    net.layerConnect(i,i) = true;
    net.layerWeights{i,i}.delays = param.layerDelays;
  end
  if isdeployed
      % Do not show training GUI for deployed code
      net.trainParam.showWindow = false;
  else
      % Add plot functions if code is not deployed
      net.plotFcns = [net.plotFcns {'plotresponse','ploterrcorr','plotinerrcorr'}];
  end 
end
