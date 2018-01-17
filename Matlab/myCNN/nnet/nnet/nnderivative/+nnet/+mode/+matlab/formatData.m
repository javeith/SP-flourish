function data = formatData(net,data,hints)

% Copyright 2015 The MathWorks, Inc.

  % Combine masks
  if isfield(data,'train')
    data.MASKS = { data.train.mask, data.val.mask, data.test.mask };
    data = rmfield(data,'train');
    data = rmfield(data,'val');
    data = rmfield(data,'test');
  end

  % Initially assume no precalculations or flattening
  data.doProcessInputs = true;
  data.doDelayInputs = true;
  data.doFlatten = false;
  data.originalQ = data.Q;
  data.originalTS = data.TS;
    
  % If not batching, then precalcuate values to optimize speed over memory.
  if isfinite(hints.batchSize)
    if ((net.numInputDelays + net.numLayerDelays) == 0)
      data = iFormatDataForStaticNetwork(net,data,hints);
    elseif (net.numLayerDelays == 0)
      data = iFormatDataForDynamicInputs(net,data,hints);
    else
      data = iFormatDataForDynamicLayers(net,data,hints);
    end
  end
  
  % Cast data for CPU/GPU and precision
  data.arrayType = hints.arrayType;
  data = nnet.internal.array.castAll(data,data.arrayType,...
    {'X','Xi','Xp','Xd','Ai','T','EW','MASKS'});
end

function data = iFormatDataForStaticNetwork(net,data,hints)
  % For static networks:
  % - Time can be flattened first.
  % - Then inputs can be preprocessed.
  % - No need to delay inputs as there are no input delays.
  
  % Flatten time
  data.doFlatten = (data.TS > 1);
  if data.doFlatten
    data = nnet.internal.data.flattenTime(data,{'X','T','EW','MASKS'});
    data.Q = data.originalQ * data.originalTS;
    data.TS = 1;
  end

  % Process inputs
  data.Xp = nnet.mode.matlab.processInputs(net,data.X,data.Q,hints);
  data.doProcessInputs = false;
end

function data = iFormatDataForDynamicInputs(net,data,hints)
  % For networks with input delays (but no layer delays)
  % - Combine initial input states with inputs
  % - Process the inputs
  % - Delay the inputs
  % - Now that delays have been eliminated, flatten time

  % Process input states and inputs
  Xc = [data.Xi data.X];
  Xp = nnet.mode.matlab.processInputs(net,Xc,data.Q,hints);
  data.doProcessInputs = false;

  % Delay inputs
  data.Xd = nnet.mode.matlab.delayInputs(net,Xp,data.Q,hints);
  data.doDelayInputs = false;

  % Flatten Time
  data.doFlatten = (data.TS > 1);
  if data.doFlatten
    data = nnet.internal.data.flattenTime(data,{'T','EW','MASKS'});
    data.Xd = iFlattenDelayedInputs(net,data.Xd);
    data.Q = data.originalQ * data.originalTS;
    data.TS = 1;
  end
end

function data = iFormatDataForDynamicLayers(net,data,hints)
  % For networks with layer delays
  % - Combine input delays states and inputs
  % - Preprocess inputs
  % - Delay inputs
  % - Time cannot be flattened since layer delays cannot be flattened
  
  % Process inputs
  Xc = [data.Xi data.X];
  Xp = nnet.mode.matlab.processInputs(net,Xc,data.Q,hints);
  data.doProcessInputs = false;

  % Delay inputs
  data.Xd = nnet.mode.matlab.delayInputs(net,Xp,data.Q,hints);
  data.doDelayInputs = false;
end

function Xdf = iFlattenDelayedInputs(net,Xd)
  [n,m,~] = size(Xd);
  Xdf = cell(n,m);
  for i = 1:net.numLayers
    for j = 1:net.numInputs
      Xdf{i,j} = [ Xd{i,j,:} ];
    end
  end
end
