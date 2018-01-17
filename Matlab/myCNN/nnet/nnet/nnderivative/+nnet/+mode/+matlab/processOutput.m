function [y,yp] = processOutput(y,ii,hints)
%PROCESSOUTPUT Post-process output
  
% Copyright 2015 The MathWorks, Inc.

  N = hints.numOutProc(ii);
  
  % Allocate saved outputs if needed for backpropatation of derivatives
  saveYp = (nargout >= 2);
  if saveYp
    yp = cell(1,N+1);
    yp{N+1} = y;
  end
  
  % Reverse process outputs in reverse function order
  for j = N:-1:1
    y = hints.out(ii).procRev{j}(y,hints.out(ii).procSet{j});
    
    % Accumulate saved processed outputs
    if saveYp
      yp{j} = y;
    end
  end
end
