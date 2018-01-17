function [perfs,counts,indOfNaN] = applyMasks(perfs,counts,i,ts,perf,MASKS,numMasks)
%APPLYMASKS Apply training, valiation and test masks to performance values

% Copyright 2015 The MathWorks, Inc.

  % Loop inreverse order so indOfNaN for first mask will be returned
  for k=numMasks:-1:1
    
    % Apply mask
    perfk = perf .* MASKS{k}{i,ts};
    
    % Find NaN values representing:
    % a) don't-knows/don't-cares due to NaN outputs or targets
    % b) values not included in this dataset due to NaN in MASK
    indOfNaN = find(isnan(perfk));
    
    % Accumulate known performance and count of values for this mask
    counts(k) = counts(k) + numel(perfk) - length(indOfNaN);
    perfk(indOfNaN) = 0;
    perfs(k) = perfs(k) + sum(sum(perfk));
  end
end
