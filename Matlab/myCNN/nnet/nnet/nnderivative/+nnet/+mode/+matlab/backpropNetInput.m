function dz = backpropNetInput(i,j,dn,z,n,hints)
% backpropNetInput

% Copyright 2015-2016 The MathWorks, Inc.

  % Net input info
  backpropFcn = hints.netBP{i};
  param = hints.netParam{i};
  
  % Backprop
  dz = backpropFcn(dn,j,z,n,param);
end
