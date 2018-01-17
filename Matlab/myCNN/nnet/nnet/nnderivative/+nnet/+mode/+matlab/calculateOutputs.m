function [Y,Af] = calculateOutputs(net,data,hints)
%NNET.MATLAB.CALCULATEOUTPUTS Calculate neural network outputs

% Copyright 2012-2015 The MathWorks, Inc.

  % Calculate outputs and final layer states, with or without batching
  B = hints.batchSize;
  if (B >= data.Q)
    [Y,Af] = iCalculateOutputs(net,data,hints);
  else
    [Y,Af] = iCalculateWithBatches(net,data,hints);
  end
  
  % Unflatten time
  if data.doFlatten
    Y = nnet.internal.data.unflattenTime(Y,data.originalTS);
  end
end

function [Y,Af] = iCalculateWithBatches(net,data,hints)

  Q = data.Q; % All samples
  B = hints.batchSize; % Samples per batch
  
  % Allocate results
  zero = zeros('like',hints.arrayType);
  Y = nndata(nn.output_sizes(net),Q,data.TS,zero);
  Af = nndata(nn.layer_sizes(net),Q,net.numLayerDelays,zero);

  % Iterate over batches
  for batchStart = 1:B:Q

    % Get batch of data
    % (Each batch is size B, except last batch may be less)
    batchStop = min(batchStart+B-1,Q);
    batchIndices = batchStart:batchStop;
    batchSize = batchStop - batchStart + 1;
    batch = nnet.internal.data.getBatch(data,Q,batchIndices,...
      {'X','Xi','Xp','Xd','Ai'});
    batch.Q = batchSize;

    % Calculate outputs
    [y,af] = iCalculateOutputs(net,batch,hints);

    % Accumulate results
    Y = nnet.internal.data.setBatch(Y,y,batchIndices);
    Af = nnet.internal.data.setBatch(Af,af,batchIndices);
  end
end

function [Y,Af] = iCalculateOutputs(net,data,hints)
  
  import nnet.mode.matlab.getProcessedInputStates
  import nnet.mode.matlab.getProcessedInputs
  import nnet.mode.matlab.evaluateLayer
  import nnet.mode.matlab.processOutput
  import nnet.mode.matlab.wrapTimesteps
  
  % Allocate outputs
  Y = cell(net.numOutputs,data.TS);
  
  % Only need to store enough input and layer outputs for delay states + 1
  inputTimeWindow = net.numInputDelays + 1;
  layerTimeWindow = net.numLayerDelays + 1;

  % Allocate temporary values
  Ac = [data.Ai cell(net.numLayers,1)];
  
  % Preprocess Initial Input States
  Xp = getProcessedInputStates(net,data,hints,inputTimeWindow);

  % Loop forward through time
  for ts = 1:data.TS

    % Preprocess Inputs
    Xp = getProcessedInputs(net,data,Xp,ts,hints,inputTimeWindow);
    
    % Loop forward through layers
    for i = hints.layerOrder

      % Evaluate Layer
      timeslot = wrapTimesteps(net.numLayerDelays+ts,layerTimeWindow);
      Ac{i,timeslot} = evaluateLayer(net,i,ts,data,Xp,Ac,hints,inputTimeWindow,layerTimeWindow);

      % Output Post-processing
      if net.outputConnect(i)
        ii = hints.layer2Output(i);
        Y{ii,ts} = processOutput(Ac{i,timeslot},ii,hints);
      end
    end
  end

  % Final Layer States
  if nargout > 1
    timesteps = wrapTimesteps((1:net.numLayerDelays)+data.TS,layerTimeWindow);
    Af = Ac(:,timesteps);
  end
end

