function y = flattenTime(x,fields)
%FLATTENTIME Flatten time by converting timesteps to samples

% Copyright 2015-2016 The MathWorks, Inc.

  if isstruct(x)
    y = iFlattenStructure(x,fields);
  elseif iIsCellofCell(x)
    y = iFlattenMasks(x);
  elseif iscell(x)
    y = iFlattenCellArray(x);
  else
    y = x;
  end
end

function x = iFlattenStructure(x,fields)
  for i = 1:numel(fields)
    field = fields{i};
    if isfield(x,field)
      x.(field) = nnet.internal.data.flattenTime(x.(field),fields);
    end
  end
end

function y = iFlattenCellArray(x)
  [M,TS] = size(x);
  y = cell(M,1);
  if ~isempty(x)
    Q = size(x{1},2);
    for i=1:M

      % Combine timesteps into samples
      y{i} = [ x{i,:} ];

      % Handle Error Weights which may require dimension expansion
      % Perform dimension expansion if either Q or TS was equal to 1
      % but not both.
      QTS = size(y(i),2);
      if (QTS ~= 1) && (QTS ~= Q*TS)
        y(i) = repmat(y(i),1,Q*TS/QTS);
      end
    end
  end
end

function flag = iIsCellofCell(x)
  flag = iscell(x) && ~isempty(x) && iscell(x{1});
end

function y = iFlattenMasks(x)
  y = cell(size(x));
  for i=1:numel(x)
    y{i} = iFlattenCellArray(x{i});
  end
end
