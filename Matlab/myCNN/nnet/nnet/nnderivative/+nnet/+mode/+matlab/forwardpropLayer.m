function [a,da] = forwardpropLayer(net,i,ts,data,Xp,Ac,dA,hints,inputTimeWindow,layerTimeWindow)
%FORWARDPROPLAYER Forward propagate signals and derivatives through a layer

% Copyright 2016 The MathWorks, Inc.

  import nnet.mode.matlab.getDelayedInput
  import nnet.mode.matlab.applyInputWeight
  import nnet.mode.matlab.delayLayerOutput
  import nnet.mode.matlab.applyLayerWeight
 
  % Allocate Weighted Inputs and Derivatives for Biases and Weights
  numZ = hints.numZ(i);
  Z = cell(1,numZ);
  dN = zeros(net.layers{i}.size,data.Q,net.numWeightElements,'like',hints.arrayType);
  
  % Evaluate Bias
  if net.biasConnect(i)
    Z{1} = net.b{i};
  end
  
  % Evaluate Input Weights
  for j = 1:net.numInputs
    if net.inputConnect(i,j)
      
      % Delays
      Xd = getDelayedInput(net,i,j,ts,data,Xp,hints,inputTimeWindow);
      
      % Apply Weight
      zInd = hints.iwzInd(i,j);
      Z{zInd} = applyInputWeight(net,i,j,Xd,hints);
    end
  end

  % Evaluate Layer Weights
  for j = 1:net.numLayers
    if net.layerConnect(i,j)
      
      % Delays
      Ad = delayLayerOutput(net,i,j,ts,Ac,hints,layerTimeWindow);
      
      % Apply Weight
      zInd = hints.lwzInd(i,j);
      Z{zInd} = applyLayerWeight(net,i,j,Ad,hints);
    end
  end

  % Net Input Function
  n = hints.netApply{i}(Z(1:hints.numZ(i)),...
    net.layers{i}.size,data.Q,hints.netParam{i},hints.arrayType);
  
  % Forwardprop Bias Derivative -> Net Input
  if net.biasConnect(i)
    
    % Bias Derivative
    S = net.layers{i}.size;
    dz = reshape(eye(S,'like',hints.arrayType),S,1,S);
    dz = repmat(dz,1,data.Q,1);
    dn = hints.netFP{i}(dz,1,Z,n,hints.netParam{i});
    
    % Sum Net Input Derivative
    gradInd = hints.bInd{i};
    dN(:,:,gradInd) = dN(:,:,gradInd) + dn;
  end
  
  % Forwardprop Input Weight Derivatives -> Net Input
  dN = iForwardPropInputWeightDerivatives(dN,net,i,ts,data,Xp,n,Z,hints,inputTimeWindow);
  
  % Forwardprop Layer Weight Derivatives -> Net Input
  dN = iForwardPropLayerWeightDerivatives(dN,net,i,ts,data,dA,Z,n,Ac,hints,layerTimeWindow);
  
  % Transfer Function
  a = hints.tfApply{i}(n,hints.tfParam{i});
  da = hints.tfFP{i}(dN,n,a,hints.tfParam{i});
end

function dN = iForwardPropInputWeightDerivatives(dN,net,i,ts,data,Xp,n,Z,hints,inputTimeWindow)

  import nnet.mode.matlab.getDelayedInput
  
  for j = 1:net.numInputs
    if net.inputConnect(i,j)
      
      % Delays
      Xd = getDelayedInput(net,i,j,ts,data,Xp,hints,inputTimeWindow);
      
      % Weight Derivative
      zInd = hints.iwzInd(i,j);
      dz = hints.iwFS{i,j}(net.IW{i,j},Xd,Z{zInd},hints.iwParam{i,j});
      
      % Sum Net Input Derivative
      gradInd = hints.iwInd{i,j};
      dn = hints.netFP{i}(dz,zInd,Z,n,hints.netParam{i});
      dN(:,:,gradInd) = dN(:,:,gradInd) + dn(:,:,:);
    end
  end
end

function dN = iForwardPropLayerWeightDerivatives(dN,net,i,ts,data,dA,Z,n,Ac,hints,layerTimeWindow)

  import nnet.mode.matlab.delayLayerOutput
  import nnet.mode.matlab.forwardpropLayerDelays
  
  for j = 1:net.numLayers
    if net.layerConnect(i,j)
      
      % Delays
      Ad = delayLayerOutput(net,i,j,ts,Ac,hints,layerTimeWindow);
      
      % Forwardprop through delays
      dad = forwardpropLayerDelays(net,i,j,ts,dA,data.Q,hints,layerTimeWindow);
      
      % Forwardprop through weight
      zInd = hints.lwzInd(i,j);
      dz = hints.lwFP{i,j}(dad,net.LW{i,j},Ad,Z{zInd},hints.lwParam{i,j});
      
      % Add New Weight Derivative
      if hints.lwInclude(i,j)
        gradInd = hints.lwInd{i,j};
        dz2 = hints.lwFS{i,j}(net.LW{i,j},Ad,Z{zInd},hints.lwParam{i,j});
        dz(:,:,gradInd) = dz(:,:,gradInd) + dz2(:,:,:);
      end
      
      % Sum Net Input Derivative
      dN = dN + hints.netFP{i}(dz,zInd,Z,n,hints.netParam{i});
    end
  end
end
