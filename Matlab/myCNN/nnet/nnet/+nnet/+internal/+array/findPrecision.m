function p = findPrecision(x)
%NNET.ARRAY.PRECISION Get the precision from Cell, array or gpuArray

% Copyright 2015 The MathWorks, Inc.

% Cell
if iscell(x)
  if isempty(x)
    p = '';
  else
    p = nnet.internal.array.findPrecision(x{1});
  end

% gpuArray
elseif isa(x,'gpuArray')
  p = classUnderlying(x);
  
% Other
else
  p = class(x);
end
