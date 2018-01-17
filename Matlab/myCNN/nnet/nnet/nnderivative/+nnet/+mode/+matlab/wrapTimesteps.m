function ts = wrapTimesteps(ts,windowSize)
%NNET.MODE.MATLAB.wrapTimesteps Wrap timesteps within timestep window

% Copyright 2015 The MathWorks, Inc.

  ts = rem(ts-1,windowSize)+1;
end
