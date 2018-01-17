function type = findType(x,precision,useGPU,fields)
%FINDTYPE Get single/double cpu/gpuArray type from array
%  The array type is represented with an empty array of that type

% Copyright 2015 The MathWorks, Inc.

  % Default arugments
  if (nargin < 2) || isempty(precision)
    precision = 'data';
  end
  if (nargin < 3) || isempty(useGPU)
    useGPU = 'data';
  end
  if (nargin < 4)
    fields = cell(1,0);
  end

  % Get type information by traversing data
  [anySingle,anyDouble,anyGPU] = iTraverse(x,false,false,false,fields);
  
  % Replace Defaults
  [useGPU,precision] = iOverrideDefaults(precision,useGPU,anySingle,anyDouble,anyGPU);
  
  % Create reference type value
  type = nnet.internal.array.createType(useGPU,precision);
end

% Traverse structures, cell arrays, and individual arrays
function [anySingle,anyDouble,anyGPU] = iTraverse(x,anySingle,anyDouble,anyGPU,fields)
  if isstruct(x)
    [anySingle,anyDouble,anyGPU] = iStructure(x,anySingle,anyDouble,anyGPU,fields);
  elseif iscell(x)
    [anySingle,anyDouble,anyGPU] = iCellArray(x,anySingle,anyDouble,anyGPU,fields);
  else
    [anySingle,anyDouble,anyGPU] = iSingleArray(x,anySingle,anyDouble,anyGPU);
  end
end
  
% Detect any gpuArray or single precision
function [anySingle,anyDouble,anyGPU] = iSingleArray(x,anySingle,anyDouble,anyGPU)
  % Any gpuArray?
  if isa(x,'gpuArray')
    anyGPU = true;
  end
  % Any Single or Double?
  if isa(x,'gpuArray')
    c = classUnderlying(x);
  else
    c = class(x);
  end
  if strcmp(c,'single')
    anySingle = true;
  else
    anyDouble = true;
  end
end

% Traverse elements of cell array
function [anySingle,anyDouble,anyGPU] = iCellArray(x,anySingle,anyDouble,anyGPU,fields)
  for i=1:numel(x)
    [anySingle,anyDouble,anyGPU] = iTraverse(x{i},anySingle,anyDouble,anyGPU,fields);
  end
end

% Traverse specified fields of a data structure
function [anySingle,anyDouble,anyGPU] = iStructure(x,anySingle,anyDouble,anyGPU,fields)
  n = numel(fields);
  for i=1:n
    field = fields{i};
    if isfield(x,field)
      [anySingle,anyDouble,anyGPU] = iTraverse(x.(field),anySingle,anyDouble,anyGPU,fields);
    end
  end
end

% Override defaults based on data properties
function [useGPU,precision] = iOverrideDefaults(precision,useGPU,anySingle,anyDouble,anyGPU);
  if strcmp(precision,'data')
    if anySingle
      precision = 'single';
    else
      precision = 'double';
    end
  end
  if strcmp(useGPU,'data')
    if anyGPU
      useGPU = 'yes';
    else
      useGPU = 'no';
    end
  elseif strcmp(useGPU,'only')
    useGPU = 'yes';
  end
end
