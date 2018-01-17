function [je,jj,perfs,counts] = calculateJacobianForwardprop(net,data,hints,numMasks)
%calculateJacobianForwardprop

% Copyright 2015-2016 The MathWorks, Inc.

  import nnet.mode.matlab.getProcessedInputStates
  import nnet.mode.matlab.getProcessedInputs
  import nnet.mode.matlab.processOutput
  import nnet.mode.matlab.forwardpropLayer
  import nnet.mode.matlab.forwardpropOutput
  import nnet.mode.matlab.applyObservationWeights
  import nnet.mode.matlab.applyMasks
  import nnet.mode.matlab.wrapTimesteps
  
  % Allocate outputs
  je = zeros(net.numWeightElements,1,'like',hints.arrayType);
  jj = zeros(net.numWeightElements,net.numWeightElements,'like',hints.arrayType);
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

        % Perf
        perf = hints.perfApply(t,y,e,hints.perfParam);
        [perf,ew] = applyObservationWeights(perf,data.EW,ii,ts,hints);
        
        % Error and Derivative
        % Divide effect of observation weights between e and dy
        if hints.doEW
          sqrtew = sqrt(ew);
          e = bsxfun(@times,e,sqrtew);
          dy = bsxfun(@times,dy,sqrtew);
        end
        
        % Masks
        [perfs,counts,indOfNaN] = applyMasks(perfs,counts,ii,ts,perf,data.MASKS,numMasks);
        e(indOfNaN) = 0;
        jac = reshape(dy,hints.outputSizes(ii)*data.Q,net.numWeightElements);
        jac(indOfNaN,:) = 0;
        
        % Accumulate Jacobian values
        % Jacobian * error and Jacobian' * Jacobian
        je = je - (e(:)' * jac)';
        jj = jj + jac' * jac;
      end
    end
  end
end
