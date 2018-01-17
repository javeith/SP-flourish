function y = getBatch(x,Q,indices,fields)
%GETBATCH Get a batch of samples from data

% Copyright 2015 The MathWorks, Inc.

  if isstruct(x)
    y = iGetBatchStructure(x,Q,indices,fields);
  elseif iscell(x)
    y = iGetBatchCellArray(x,Q,indices);
  else
    y = iGetBatchArray(x,Q,indices);
  end
end

function x = iGetBatchStructure(x,Q,indices,fields)
  % Get columns from each field of x
  for i = 1:numel(fields)
    field = fields{i};
    if isfield(x,field)
      x.(field) = nnet.internal.data.getBatch(x.(field),Q,indices,fields);
    end
  end
end

function x = iGetBatchCellArray(x,Q,indices)
  % Get columns from each element of x
  for i=1:numel(x)
    x{i} = nnet.internal.data.getBatch(x{i},Q,indices);
  end
end

function x = iGetBatchArray(x,Q,indices)
  % Get columns of array x
  % (Unless x only has 1 column and Q is ~= 1 which means x will be
  % column expanded later, so should remain unchanged here.)
  if (size(x,2) ~= 1) || (Q == 1)
    x = x(:,indices);
  end
end
