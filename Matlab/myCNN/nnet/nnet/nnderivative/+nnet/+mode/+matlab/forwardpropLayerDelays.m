function dad = forwardpropLayerDelays(net,i,j,ts,dA,Q,hints,layerTimeWindow)
%FORWARDPROPLAYERDELAYS Pass derivatives forward through layer delays

% Copyright 2016 The MathWorks, Inc.

  % Shift delays to current time index
  timesteps = net.numLayerDelays - net.layerWeights{i,j}.delays + ts;
  
  % Take modulus of timesteps into layer time window
  if (nargin >= 7)
    timesteps = nnet.mode.matlab.wrapTimesteps(timesteps,layerTimeWindow);
  end
  
  % Combine inputs across the delay timesteps
  if isempty(timesteps)
    % 0 timesteps, empty matrix
    dad = zeros(0,Q,'like',hints.arrayType);
  elseif isscalar(timesteps)
    % 1 timestep, select
    dad = dA{j,timesteps};
  else
    % >1 timestep, concatenate
    dad = cat(1,dA{j,timesteps});
  end
end
