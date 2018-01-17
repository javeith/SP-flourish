function y = unflattenTime(x,TS,fields)
%UNFLATTENTIME Unflatten time by converting samples to timesteps

% Copyright 2015 The MathWorks, Inc.

  if isstruct(x)
    y = iUnflattenStructure(x,TS,fields);
  elseif iscell(x)
    y = iUnflattenCellArray(x,TS);
  else
    y = x;
  end
end

function x = iUnflattenStructure(x,TS,fields)
  for i = 1:numel(fields)
    field = fields{i};
    if isfield(x,field)
      x.(field) = nnet.internal.data.unflattenTime(x.(field),TS,fields);
    end
  end
end

function y = iUnflattenCellArray(x,TS)

  % No sample->time change necessary
  if (size(x,2) == TS)
    y = x;
    
  % Split each array in X into TS parts
  else
    M = size(x,1);
    y = cell(M,TS);
    for i=1:M
      xi = x{i};
      Q1 = size(xi,2);
      Q2 = Q1 / TS;
      for ts=1:TS
        q = (1:Q2) + (ts-1)*Q2;
        y{i,ts} = xi(:,q);
      end
    end
  end
end
