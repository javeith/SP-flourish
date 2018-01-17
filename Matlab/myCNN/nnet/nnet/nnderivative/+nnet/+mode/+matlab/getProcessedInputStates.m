function xp = getProcessedInputStates(net,data,hints,inputTimeWindow)
%getProcessedInputStates Calculate processed input states

% Copyright 2015 The MathWorks, Inc.

  % Only need to calculate if not precalculated
  if data.doProcessInputs
    xp = cell(net.numInputs,inputTimeWindow);
    for ts = 1:net.numInputDelays
      for i = 1:net.numInputs
        xp{i,ts} = nnet.mode.matlab.processInput(data.Xi{i,ts},i,hints);
      end
    end
    
  else
    xp = {};
  end
end
