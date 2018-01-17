function dw = backstopLayerWeightJacobian(net,i,j,a,z,dz,hints)
%BACKSTOPLAYERWEIGHTJACOBIAN Backprop derivatives to individual weight elements

% Copyright 2015-2016 The MathWorks, Inc.

  % Weight info
  w = net.LW{i,j};
  backstopParallelFcn = hints.lwBSP{i,j};
  param = hints.lwParam{i,j};
  
  % Backstop
  dw = backstopParallelFcn(dz,w,a,z,param);
end
