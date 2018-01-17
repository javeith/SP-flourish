function [perf,ew] = applyObservationWeights(perf,EW,i,ts,hints)
%APPLYOBSERVATIONWEIGHTS Apply observation weights to performance values

% Copyright 2015 The MathWorks, Inc.

  % If weights are to be scalar expanded across signals, set i to 1
  if (hints.M_EW == 1)
    i = 1;
  end
    
  % Apply observation weights to performance
  ew = EW{i,ts};
  perf = bsxfun(@times,perf,ew);
end
