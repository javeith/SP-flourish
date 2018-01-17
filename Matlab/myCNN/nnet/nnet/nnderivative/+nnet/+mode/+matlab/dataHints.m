function hints = dataHints(data,hints)

% Copyright 2012-2015 The MathWorks, Inc.

% Error Weight hints
if isfield(data,'T')
  hints.doEW = any(any(nnet.internal.array.cell2Mat(data.EW) ~= 1));
  try
    hints.doEW = gather(hints.doEW);
  catch
    % Gather not defined
  end
  [hints.N_EW,hints.Q_EW,hints.TS_EW,hints.M_EW] = nnsize(data.EW);
else
  hints.doEW = false;
end

% Create reference value, for use with cast(), zeros(), ones(), rand(), etc.
hints.arrayType = nnet.internal.array.findType(data,hints.precision,hints.useGPU,...
  {'X','Xi','Ai','T'});

% Update useGPU and precision hints to match reference array type
if isa(hints.arrayType,'gpuArray')
  hints.useGPU = true;
  hints.precision = classUnderlying(hints.arrayType);
else
  hints.useGPU = false;
  hints.precision = class(hints.arrayType);
end
