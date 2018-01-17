function x = setBatch(x,batch,indices)
%SETBATCH Set a batch of samples in data

% Copyright 2015 The MathWorks, Inc.

  if iscell(x)
    x = iSetBatchCellArray(x,batch,indices);
  else
    x = iSetBatchArray(x,batch,indices);
  end
end

function x = iSetBatchCellArray(x,batch,indices)
  % Set columns of each element of cell array x
  for i=1:numel(x)
    x{i} = iSetBatchArray(x{i},batch{i},indices);
  end
end

function x = iSetBatchArray(x,batch,indices)
  % Set columns of array x
  x(:,indices) = batch;
end
