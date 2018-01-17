function da = backpropLayerWeight(net,i,j,dz,a,z,hints)
%BACKPROPLAYERWEIGHT Backpropagate derivatives through layer weight

% Copyright 2015-2016 The MathWorks, Inc.

  % Weight info
  w = net.LW{i,j};
  backpropFcn = hints.lwBP{i,j};
  param = hints.lwParam{i,j};
  
  % Backprop
  da = backpropFcn(dz,w,a,z,param);
end
