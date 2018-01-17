function Xp = getProcessedInputs(net,data,Xp,ts,hints,inputWindowSize)
%getProcessedInputs Calculate or select processed inputs

% Copyright 2015 The MathWorks, Inc.

  if data.doProcessInputs
    Xp = iCalculate(net,data,Xp,ts,hints,inputWindowSize);
  elseif data.doDelayInputs
    Xp = iSelectPrecalculatedValues(net,data,Xp,ts,inputWindowSize);
  else
    Xp = []; % Not needed
  end
end

% Calculate processed inputs
function xp = iCalculate(net,data,xp,ts,hints,inputWindowSize)
  for i = 1:net.numInputs
    timesteps = rem(net.numInputDelays-1+ts,inputWindowSize)+1;
    xp{i,timesteps} = nnet.mode.matlab.processInput(data.X{i,ts},i,hints);
  end
end

% Get pre-calculated processed inputs
function xp = iSelectPrecalculatedValues(net,data,xp,ts,inputWindowSize)
  timesteps = rem((0:net.numInputDelays)+ts-1,inputWindowSize)+1;
  xp(:,timesteps) = data.Xp(:,(0:net.numInputDelays)+ts);
end
