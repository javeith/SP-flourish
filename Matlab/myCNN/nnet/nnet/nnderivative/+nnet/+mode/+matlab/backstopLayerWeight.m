function dw = backstopLayerWeight(net,i,j,a,z,dz,hints)
%BACKSTOPLAYERWEIGHT Backpropagate derivatives to a layer weight

% Copyright 2015-2016 The MathWorks, Inc.

  % Weight info
  w = net.LW{i,j};
  backstopFcn = hints.lwBS{i,j};
  param = hints.lwParam{i,j};
  
  % Backstop
  dw = backstopFcn(dz,w,a,z,param);
end
