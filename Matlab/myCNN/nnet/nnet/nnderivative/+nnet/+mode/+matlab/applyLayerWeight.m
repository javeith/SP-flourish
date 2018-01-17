function z = applyLayerWeight(net,i,j,Ad,hints)
%APPLYLAYERWEIGHT Apply layer weights and function

% Copyright 2015 The MathWorks, Inc.

  % Weight, function and paramaters
  w = net.LW{i,j};
  weightFcn = hints.lwApply{i,j};
  weightFcnParam = hints.lwParam{i,j};
  
  % Apply
  z = weightFcn(w,Ad,weightFcnParam);
end
