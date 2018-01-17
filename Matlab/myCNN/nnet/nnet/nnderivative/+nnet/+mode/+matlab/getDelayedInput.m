function xd = getDelayedInput(net,i,j,ts,data,Xp,hints,inputTimeWindow)
%getProcessedInputStates Calculate or select delayed inputs

% Copyright 2015 The MathWorks, Inc.

  % Only need to calculate if not precalculated
  if data.doDelayInputs
    if (nargin < 8)
      inputTimeWindow = net.numInputDelays + data.TS;
    end
    xd = nnet.mode.matlab.delayInput(net,i,j,ts,Xp,hints,inputTimeWindow);
  else
    xd = data.Xd{i,j,ts};
  end
end
