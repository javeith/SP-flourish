function dw = backstopInputWeight(net,i,j,x,z,dz,hints)
%BACKSTOPINPUTWEIGHT   Backprop derivatives to input weight

% Copyright 2015-2016 The MathWorks, Inc.

  % Weight info
  w = net.IW{i,j};
  backstopFcn = hints.iwBS{i,j};
  param = hints.iwParam{i,j};
  
  % Backstop
  dw = backstopFcn(dz,w,x,z,param);
end
