function [gpu,precision] = getTypeInfo(x) 
%GETTYPEINFO Get GPU and precision status of a single array.

% Copyright 2015 The MathWorks, Inc.

  if isa(x,'gpuArray')
    gpu = 'yes';
    x = gather(x);
  else
    gpu = 'no';
  end
  
  precision = class(x);
end
