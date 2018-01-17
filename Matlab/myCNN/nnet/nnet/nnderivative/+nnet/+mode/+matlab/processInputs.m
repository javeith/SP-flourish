function Xp = processInputs(net,X,Q,hints)
%NNET.MODE.MATLAB.PROCESSINPUTS Preprocess all of a neural networks inputs

% Copyright 2012-2016 The MathWorks, Inc.

  TS = size(X,2);
  Xp = cell(net.numInputs,TS);

  if (TS > 0)
    for i = 1:net.numInputs
      x = [X{i,:}];
      xp = nnet.mode.matlab.processInput(x,i,hints);
      R2 = size(xp,1);
      Xp(i,:) = mat2cell(xp,R2,zeros(1,TS)+Q);
    end
  end
end
