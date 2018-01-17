function dA = backpropLayerDelays(net,i,j,ts,dA,dAd)
%BACKPROPLAYERDELAYS Backpropagate derivatives through layer delays

% Copyright 2015-2016 The MathWorks, Inc.

  % Timesteps to backprop to
  timesteps = ts - net.layerWeights{i,j}.delays;
  numTimesteps = numel(timesteps);
  
  % Previous layer size being backpropagated to
  layerSize = size(dAd,1) / numTimesteps;
  
  if isscalar(timesteps)
    % Backpropagate through single delay
    % (No need to divide up matrix)
    dA{j,timesteps} = iAdd(dA{j,timesteps},dAd);
    
  else
    % Backpropagate to each timestep
    for k=1:numTimesteps
      timestep = timesteps(k);

      % Backpropagate for time >= 1
      % (Don't backpropagate to negative time)
      if (timestep >= 1)
        indices = (layerSize*(k-1)) + (1:layerSize);
        dA{j,timestep} = iAdd(dA{j,timestep},dAd(indices,:));
      end
    end
  end
end

function c = iAdd(a,b)
% Add B to possibly empty matrix A
  if isempty(a)
    c = b;
  else
    c = a + b;
  end
end

