function Q = findQ(varargin)
%NNET.INTERNAL.DATA.FINDQ Determine number of samples from data

% Q is number of columns in first non-empty (non-cell) array.
% If all cell arrays are empty, Q is 0.

% Copyright 2015 The MathWorks, Inc.

  Q = [];
  for i=1:nargin
    Q = iFindQ_1Argument(varargin{i});
    if ~isempty(Q)
      break;
    end
  end
  if isempty(Q)
    Q = 0;
  end
end

function Q = iFindQ_1Argument(x)
    
  % Cell array
  if iscell(x)
    
    Q = [];
    for i = 1:numel(x)
      Q = iFindQ_1Argument(x{i});
      if ~isempty(Q)
        break;
      end
    end

  % Single Array
  else
    Q = size(x,2);
  end
end
