function [gWB,perfs,counts] = calculateGradientBackprop(net,data,hints,numMasks)
%calculateGradientBackprop Backpropagate performance gradient to weights

% Copyright 2016 The MathWorks, Inc.

  import nnet.mode.matlab.getProcessedInputStates
  import nnet.mode.matlab.getProcessedInputs
  import nnet.mode.matlab.evaluateLayer
  import nnet.mode.matlab.processOutput
  import nnet.mode.matlab.calculateOutputPerformance
  import nnet.mode.matlab.backpropLayer
  import nnet.mode.matlab.wrapTimesteps
  
  % Allocate outputs
  [perfs,counts,dB,dIW,dLW] = iAllocateValues(net,hints,numMasks);
  
  % For backprop gradient the full time windows are needed
  inputTimeWindow = net.numInputDelays + data.TS;
  layerTimeWindow = net.numLayerDelays + data.TS;

  % Allocate Temporary Values
  Ac = [data.Ai cell(net.numLayers,data.TS)];
  N = cell(net.numLayers,data.TS);
  Z = cell(net.numLayers,hints.maxZ,data.TS);
  dA = cell(net.numLayers,data.TS);

  % Preprocess Initial Input States
  Xp = getProcessedInputStates(net,data,hints,inputTimeWindow);

  % Evaluate Forward in Time
  for ts = 1:data.TS

    % Preprocess Inputs
    Xp = getProcessedInputs(net,data,Xp,ts,hints,inputTimeWindow);

    % Layers
    for i = hints.layerOrder

      timeslot = net.numLayerDelays + ts;
      [a,n,z] = evaluateLayer(net,i,ts,data,Xp,Ac,hints,inputTimeWindow,layerTimeWindow);
      [Ac{i,timeslot},N{i,ts},Z(i,1:hints.numZ(i),ts)] = deal(a,n,z);
      
      % Outputs
      if net.outputConnect(i)
        
        % Output Post-processing
        ii = hints.layer2Output(i);
        [y,Yp] = processOutput(Ac{i,timeslot},ii,hints);
        
        t = data.T{ii,ts};
        [perfs,counts,indOfNaN,ew,e] = calculateOutputPerformance(...
          perfs,counts,ii,ts,t,y,data.EW,data.MASKS,numMasks,hints);
        
        % Derivative of Performances
        dy = -hints.perfBP(t,y,e,hints.perfParam);
        if hints.doErrNorm(ii)
          dy = bsxfun(@times,dy,hints.errNorm{ii});
        end
        if hints.doEW
          dy = bsxfun(@times,dy,ew);
        end
        dy(indOfNaN) = 0;
        
        % Backprop through Output Processing
        for j = 1:hints.numOutProc(ii)
          dy = hints.out(ii).procBPrev{j}(dy,Yp{j},Yp{j+1},hints.out(ii).procSet{j});
        end
          
        % Backprop to layer
        dA{i,ts} = dy;
      end
    end
  end

  % Gradient Backward in Time
  [dB,dIW,dLW] = iBackpropLayers(dB,dIW,dLW,dA,net,data,Xp,Z,N,Ac,hints);

  % Combine Weight and Bias Derivatives
  gWB = formwb(net,dB,dIW,dLW,hints,data.arrayType);
end

function [dB,dIW,dLW] = iBackpropLayers(dB,dIW,dLW,dA,net,data,Xp,Z,N,Ac,hints)

  import nnet.mode.matlab.backpropLayer

  % Backprop through time
  for ts = data.TS:-1:1

    % Backprop through layers
    for i = hints.layerOrderReverse
      if ~isempty(dA{i,ts})
        [dA,dB,dIW,dLW] = ...
          backpropLayer(dA,dB,dIW,dLW,net,i,ts,data,Xp,Z,N,Ac,hints);
      end
    end
  end
end

function [perfs,counts,dB,dIW,dLW] = iAllocateValues(net,hints,numMasks)

  perfs = zeros(1,numMasks,'like',hints.arrayType);
  counts = zeros(1,numMasks,'like',hints.arrayType);
  
  dB = cell(net.numLayers,1);
  dIW = cell(net.numLayers,net.numInputs);
  dLW = cell(net.numLayers,net.numLayers);
  for i=1:net.numLayers
    if net.biasConnect(i)
      dB{i} = zeros(net.layers{i}.size,1,'like',hints.arrayType);
    end
    for j=1:net.numInputs
      if net.inputConnect(i,j)
        dIW{i,j} = zeros(net.inputWeights{i,j}.size,'like',hints.arrayType);
      end
    end
    for j=1:net.numLayers
      if net.layerConnect(i,j)
        dLW{i,j} = zeros(net.layerWeights{i,j}.size,'like',hints.arrayType);
      end
    end
  end
end