function [a,n,z] = evaluateLayer(net,i,ts,data,Xp,Ac,hints,inputTimeWindow,layerTimeWindow)
%EVALUATELAYER Evaluate a neural network layer.

% Copyright 2015 The MathWorks, Inc.

  import nnet.mode.matlab.getDelayedInput
  import nnet.mode.matlab.applyInputWeight
  import nnet.mode.matlab.delayLayerOutput
  import nnet.mode.matlab.applyLayerWeight
  
  % Allocate Weighted Inputs for Bias and Weights
  z = cell(1,hints.numZ(i));
  
  % Bias
  if net.biasConnect(i)
    z{1} = net.b{i};
  end
  
  % Input Weights
  for j = 1:net.numInputs
    if net.inputConnect(i,j)
      
      % Delays
      Xd = getDelayedInput(net,i,j,ts,data,Xp,hints,inputTimeWindow);
      
      % Apply Weight
      zInd = hints.iwzInd(i,j);
      z{zInd} = applyInputWeight(net,i,j,Xd,hints);
    end
  end

  % Layer Weights
  for j = 1:net.numLayers
    if net.layerConnect(i,j)
      
      % Delays
      Ad = delayLayerOutput(net,i,j,ts,Ac,hints,layerTimeWindow);
      
      % Apply Weight
      zInd = hints.lwzInd(i,j);
      z{zInd} = applyLayerWeight(net,i,j,Ad,hints);
    end
  end

  % Apply Net Input Function
  n = hints.netApply{i}(z(1:hints.numZ(i)),...
    net.layers{i}.size,data.Q,hints.netParam{i},hints.arrayType);

  % Apply Transfer Function
  a = hints.tfApply{i}(n,hints.tfParam{i});
end
