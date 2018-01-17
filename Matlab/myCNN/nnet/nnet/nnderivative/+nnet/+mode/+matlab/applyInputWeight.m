function z = applyInputWeight(net,i,j,Xd,hints)
%APPLYINPUTWEIGHT Apply input weight and function

% Copyright 2015 The MathWorks, Inc.

  % Weight info
  w = net.IW{i,j};
  weightFcn = hints.iwApply{i,j};
  weightFcnParam = hints.iwParam{i,j};
  
  % Apply
  z = weightFcn(w,Xd,weightFcnParam);
end
