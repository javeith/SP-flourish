function dw = backstopInputWeightJacobian(net,i,j,x,z,dz,hints)
%BACKSTOPINPUTWEIGHTJACOBIAN Backprop derivatives to individual weight elements

% Copyright 2015-2016 The MathWorks, Inc.

  % Weight info
  w = net.IW{i,j};
  backstopParallelFcn = hints.iwBSP{i,j};
  param = hints.iwParam{i,j};
  
  % Backstop
  dw = backstopParallelFcn(dz,w,x,z,param);
end
