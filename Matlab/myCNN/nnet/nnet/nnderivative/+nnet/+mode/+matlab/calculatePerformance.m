function varargout = calculatePerformance(net,data,hints)
%NNET.MATLAB.CALCULATEPERFORMANCE

% Copyright 2012-2015 The MathWorks, Inc.

  % 1 or 3 Masks
  if (nargout <= 2)
    numMasks = 1;
  else
    numMasks = 3;
  end

  % Calculate performances and counts, with or without batching
  if isnan(hints.batchSize) || (hints.batchSize == data.Q)
    [perfs,counts] = iCalculatePerformances(net,data,hints,numMasks);
  else
    [perfs,counts] = iCalculateWithBatches(net,data,hints,numMasks);
  end
  
  % Cast to CPU double
  perfs = cast(perfs,'like',1);
  counts = cast(counts,'like',1);
  
  % Output arguments
  varargout = num2cell([perfs counts]);
end

function [perfs,counts] = iCalculateWithBatches(net,data,hints,numMasks)

  Q = data.Q; % All samples
  B = hints.batchSize; % Samples per batch
  
  % Allocate outputs
  perfs = zeros(1,numMasks,'like',hints.arrayType);
  counts = zeros(1,numMasks,'like',hints.arrayType);
  
  % Iterate over batches
  for batchStart = 1:B:Q

    % Get batch of data
    % (Each batch is size B, except last batch may be less)
    batchStop = min(batchStart+B-1,Q);
    batchIndices = batchStart:batchStop;
    batchSize = batchStop - batchStart + 1;
    batch = nnet.internal.data.getBatch(data,Q,batchIndices,...
      {'X','Xi','Xp','Xd','Ai','T','EW','MASKS'});
    batch.Q = batchSize;

    % Calculate performance
    [p,c] = iCalculatePerformances(net,batch,hints,numMasks);

    % Accumulate results
    perfs = perfs + p;
    counts = counts + c;
  end
end

function [perfs,counts] = iCalculatePerformances(net,data,hints,numMasks)
  
  import nnet.mode.matlab.getProcessedInputStates
  import nnet.mode.matlab.getProcessedInputs
  import nnet.mode.matlab.evaluateLayer
  import nnet.mode.matlab.processOutput
  import nnet.mode.matlab.applyObservationWeights
  import nnet.mode.matlab.applyMasks
  import nnet.mode.matlab.wrapTimesteps
  
  % Allocate outputs
  perfs = zeros(1,numMasks,'like',hints.arrayType);
  counts = zeros(1,numMasks,'like',hints.arrayType);
  
  % Only need to store enough input and layer output for delays + 1
  inputTimeWindow = net.numInputDelays + 1;
  layerTimeWindow = net.numLayerDelays + 1;

  % Allocate layer output window and outputs
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

      % Outputs
      if net.outputConnect(i)
        
        % Output Post-processing
        ii = hints.layer2Output(i);
        y = processOutput(Ac{i,timeslot},ii,hints);
        
        % Error
        t = data.T{ii,ts};
        e = t - y;
        
        % Error Normalization
        if hints.doErrNorm(ii)
          e = bsxfun(@times,e,hints.errNorm{ii});
        end
        
        % Performance
        perf = hints.perfApply(t,y,e,hints.perfParam);

        % Performance Weights
        perf = applyObservationWeights(perf,data.EW,ii,ts,hints);

        % Masks
        [perfs,counts] = applyMasks(perfs,counts,ii,ts,perf,data.MASKS,numMasks);
      end
    end
  end
end

