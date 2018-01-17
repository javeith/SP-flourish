function [dA,dB,dIW,dLW] = backpropLayer(dA,dB,dIW,dLW,net,i,ts,data,Xp,Z,N,Ac,hints)
%BACKPROPLAYER Backpropagate derivatives through a layer

% Copyright 2015-2016 The MathWorks, Inc.

  import nnet.mode.matlab.backpropNetInput
  import nnet.mode.matlab.getDelayedInput
  import nnet.mode.matlab.backstopInputWeight
  import nnet.mode.matlab.delayLayerOutput
  import nnet.mode.matlab.backstopLayerWeight
  import nnet.mode.matlab.backpropLayerWeight
  import nnet.mode.matlab.backpropLayerDelays

  % Transfer Function
  a_ts = net.numLayerDelays + ts;
  dn = hints.tfBP{i}(dA{i,ts},N{i,ts},Ac{i,a_ts},hints.tfParam{i});
  Zi = Z(i,1:hints.numZ(i),ts);

  % Bias
  if hints.bInclude(i)

    % Net Input to Expanded Bias
    dz = backpropNetInput(i,1,dn,Zi,N{i,ts},hints);

    % to Bias
    dB{i} = dB{i} + sum(dz,2);
  end
  
  % Input Weights
  for j = net.numInputs:-1:1
    if hints.iwInclude(i,j)
      zInd = hints.iwzInd(i,j);

      % Net Input to Weighted Input
      dz = backpropNetInput(i,zInd,dn,Zi,N{i,ts},hints);

      % To input weight
      Xd = getDelayedInput(net,i,j,ts,data,Xp,hints);
      dIW{i,j} = dIW{i,j} + backstopInputWeight(net,i,j,Xd,Z{i,zInd,ts},dz,hints);
    end
  end
  
  % Layer Weights
  for j = net.numLayers:-1:1
    if net.layerConnect(i,j)
      zInd = hints.lwzInd(i,j);

      % Net Input to Weighted Layer Output
      dz = backpropNetInput(i,zInd,dn,Zi,N{i,ts},hints);

      % Delays
      Ad = delayLayerOutput(net,i,j,ts,Ac,hints);

      % To Layer Weight
      if hints.lwInclude(i,j)
        dLW{i,j} = dLW{i,j} + backstopLayerWeight(net,i,j,Ad,Z{i,zInd,ts},dz,hints);
      end

      % Through Layer Weight
      dAd = backpropLayerWeight(net,i,j,dz,Ad,Z{i,zInd,ts},hints);

      % Through Delays to Previous Layers
      dA = backpropLayerDelays(net,i,j,ts,dA,dAd);
    end
  end
end
