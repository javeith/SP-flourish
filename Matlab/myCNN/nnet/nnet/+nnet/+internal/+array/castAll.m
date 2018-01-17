function x = castAll(x,arrayType,fields)
%NNET.ARRAY.CASTALL Cast array data within cell arrays and structures

% Copyright 2015-2016 The MathWorks, Inc.

  if (nargin < 3)
    fields = {};
  end

  if isstruct(x)
    x = iStructure(x,arrayType,fields);
  elseif iscell(x)
    x = iCellArray(x,arrayType,fields);
  else
    x = iEnsureFull(x);
    x = cast(x,'like',arrayType);
  end
end

% Cast elements of cell array
function x = iCellArray(x,arrayType,fields)
  f = @(x) nnet.internal.array.castAll(x,arrayType,fields);
  x = cellfun(f,x,'UniformOutput',false);
end

% Cast specified fields of structure
function x = iStructure(x,arrayType,fields)
  n = numel(fields);
  for i=1:n
    field = fields{i};
    if isfield(x,field)
      x.(field) = nnet.internal.array.castAll(x.(field),arrayType,fields);
    end
  end
end

function X = iEnsureFull( X )
if issparse( X )
    warning( message( 'nnet:NNData:ConvertSparseToFull' ) );
    X = full( X );
end
end