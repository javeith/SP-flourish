function [y,dy] = forwardpropOutput(y,dy,ii,hints)
%FORWARDPROPOUTPUT Propagate signal and derivatives through output processing
  
% Copyright 2015-2016 The MathWorks, Inc.

  N = hints.numOutProc(ii);
  
  % Reverse process outputs in reverse function order
  for j = N:-1:1
    x = y;
    y = hints.out(ii).procRev{j}(y,hints.out(ii).procSet{j});
    dy = hints.out(ii).procFPrev{j}(dy,y,x,hints.out(ii).procSet{j});
  end
end
