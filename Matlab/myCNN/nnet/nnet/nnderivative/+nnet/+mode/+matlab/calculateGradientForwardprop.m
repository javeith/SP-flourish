function [gWB,perfs,counts] = calculateGradientForwardprop(net,data,hints,numMasks)
%calculateGradientForwardprop Forward propagate weight gradient to performance

% Copyright 2015-2016 The MathWorks, Inc.

  import nnet.mode.matlab.getProcessedInputStates
  import nnet.mode.matlab.getProcessedInputs
  import nnet.mode.matlab.evaluateLayer
  import nnet.mode.matlab.processOutput
  import nnet.mode.matlab.forwardpropLayer
  import nnet.mode.matlab.forwardpropOutput
  import nnet.mode.matlab.applyObservationWeights
  import nnet.mode.matlab.applyMasks
  import nnet.mode.matlab.wrapTimesteps
  
  % Allocate outputs
  gWB = zeros(net.numWeightElements,1,'like',hints.arrayType);
  perfs = zeros(1,numMasks,'like',hints.arrayType);
  counts = zeros(1,numMasks,'like',hints.arrayType);
  
  % Only need to store enough input and layer outputs for delay states + 1
  inputTimeWindow = net.numInputDelays + 1;
  layerTimeWindow = net.numLayerDelays + 1;
  
  % Allocate temporary variables
  Ac = [data.Ai cell(net.numLayers,1)];
  dA = cell(net.numLayers,net.numLayerDelays+1);
  for i=1:net.numLayers
    dA(i,1:net.numLayerDelays) = ...
      { zeros(net.layers{i}.size,data.Q,net.numWeightElements,'like',data.arrayType) };
  end

  % Preprocess Initial Input States
  Xp = getProcessedInputStates(net,data,hints,inputTimeWindow);

  % Loop forward through time
  for ts = 1:data.TS

    % Preprocess Inputs
    Xp = getProcessedInputs(net,data,Xp,ts,hints,inputTimeWindow);

    % Layers
    for i = hints.layerOrder
      
      % Evaluate and Forwardprop Layer
      timeslot = wrapTimesteps(net.numLayerDelays+ts,layerTimeWindow);
      [Ac{i,timeslot},dA{i,timeslot}] = forwardpropLayer(net,i,ts,data,Xp,Ac,dA,hints,...
        inputTimeWindow,layerTimeWindow);

      % Output
      if net.outputConnect(i)
        
        % Output Post-processing
        ii = hints.layer2Output(i);
        [y,dy] = forwardpropOutput(Ac{i,timeslot},dA{i,timeslot},ii,hints);

        % Error
        t = data.T{ii,ts};
        e = t - y;
        
        % Error Normalization
        if hints.doErrNorm(ii)
          e = bsxfun(@times,e,hints.errNorm{ii});
          dy = bsxfun(@times,dy,hints.errNorm{ii});
        end

        % Performance
        perf = hints.perfApply(t,y,e,hints.perfParam);
        dperf = -hints.perfFP(dy,t,y,e,hints.perfParam);
        
        % Observation Weights
        if hints.doEW
          [perf,ew] = applyObservationWeights(perf,data.EW,ii,ts,hints);
          dperf = bsxfun(@times,dperf,ew);
        end

        % Masks
        [perfs,counts,indOfNaN] = applyMasks(perfs,counts,ii,ts,perf,data.MASKS,numMasks);
        
        % Accumulate gradient
        S = net.outputs{i}.size;
        dperf = reshape(dperf,data.Q * S,net.numWeightElements);
        dperf(indOfNaN,:) = 0;
        dperf = sum(dperf,1)';
        gWB = gWB + dperf;
      end
    end
  end
end
