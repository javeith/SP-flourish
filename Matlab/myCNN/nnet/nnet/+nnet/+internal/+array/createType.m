function type = createType(gpu,precision)
%CREATETYPE Create a type value from GPU and precision status.
%  The array type is represented with an empty array of that type.

% Copyright 2015 The MathWorks, Inc.

  type = ones(0,0,precision);
  if strcmp(gpu,'yes')
    type = gpuArray(type);
  end
end
