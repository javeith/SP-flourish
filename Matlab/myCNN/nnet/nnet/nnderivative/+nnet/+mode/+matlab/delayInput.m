function Xd = delayInput(net,i,j,ts,Xp,hints,inputTimeWindow)
%DELAYINPUT Apply delays to inputs

% Copyright 2015 The MathWorks, Inc.

  % Shift delays to current time index
  timesteps = net.numInputDelays - net.inputWeights{i,j}.delays + ts;
  
  % Take modulus of timesteps into input time window
  if (nargin >= 7)
    timesteps = nnet.mode.matlab.wrapTimesteps(timesteps,inputTimeWindow);
  end
  
  % Combine inputs across the delay timesteps
  if isempty(timesteps)
    % 0 timesteps, empty matrix
    Xd = zeros(0,Q,'like',hints.arrayType);
  elseif isscalar(timesteps)
    % 1 timestep, select
    Xd = Xp{j,timesteps};
  else
    % >1 timestep, concatenate
    Xd = cat(1,Xp{j,timesteps});
  end
end
