function Ad = delayLayerOutput(net,i,j,ts,Ac,hints,layerTimeWindow)
%DELAYLAYEROUTPUT Apply delays to a layer's output

% Copyright 2015 The MathWorks, Inc.

  % Shift delays to current time index
  timesteps = net.numLayerDelays - net.layerWeights{i,j}.delays + ts;
  
  % Take modulus of timesteps into layer time window
  if (nargin >= 7)
    timesteps = nnet.mode.matlab.wrapTimesteps(timesteps,layerTimeWindow);
  end
  
  % Combine inputs across the delay timesteps
  if isempty(timesteps)
    % 0 timesteps, empty matrix
    Ad = zeros(0,Q,'like',hints.arrayType);
  elseif isscalar(timesteps)
    % 1 timestep, select
    Ad = Ac{j,timesteps};
  else
    % >1 timestep, concatenate
    Ad = cat(1,Ac{j,timesteps});
  end
end
