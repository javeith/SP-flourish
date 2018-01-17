function x = processInput(x,i,hints)
%PROCESSINPUT Preprocess an input

% Copyright 2015 The MathWorks, Inc.

  for j=1:hints.numInpProc(i)
    x = hints.inp(i).procApply{j}(x,hints.inp(i).procSet{j});
  end
end