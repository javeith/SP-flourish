function [perfs,counts,indOfNaN,ew,e] = calculateOutputPerformance(perfs,counts,ii,ts,t,y,EW,MASKS,numMasks,hints)
%calculatePerformance Calculate performance and related values from network targets and outputs

% Copyright 2016 The MathWorks, Inc.

  import nnet.mode.matlab.applyObservationWeights
  import nnet.mode.matlab.applyMasks
  
  % Error
  e = t - y;

  % Error Normalization
  if hints.doErrNorm(ii)
    e = bsxfun(@times,e,hints.errNorm{ii});
  end

  % Performance
  perf = hints.perfApply(t,y,e,hints.perfParam);

  % Observation Weights
  if hints.doEW
    [perf,ew] = applyObservationWeights(perf,EW,ii,ts,hints);
  else
    ew = 1;
  end
  
  % Masks
  [perfs,counts,indOfNaN] = applyMasks(perfs,counts,ii,ts,perf,MASKS,numMasks);
end
