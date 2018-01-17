function TS = findTS(varargin)
%NNET.DATA.FINDTS Determine the number of timesteps from data.

% TS is set by the first non-0x0 cell array argument.
%   If that argument is a cell array, then TS is the number of columns.
%   If that argument is a matrix, then TS is 1.
% If all arguments are empty cell arrays then TS is 0.

% Copyright 2015 The MathWorks, Inc.
  
  TS = 0; % Zero if no arguments or cell elements found.
  
  for i=1:numel(varargin)
    x = varargin{i};
    
    % TS is columns of a non-0x0 cell array
    if iscell(x) && any(size(x) ~= 0)
      TS = size(x,2);
      break;
      
    % TS is 1 for non-cell array
    elseif ~iscell(x)
      TS = 1;
      break;
    end
  end
end
