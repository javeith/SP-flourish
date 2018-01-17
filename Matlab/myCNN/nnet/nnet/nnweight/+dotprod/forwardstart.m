function dz_dw = forwardstart(w,p,z,param)
%DOTPROD.FORWARDSTART Propagate derivative from weight to weighted input

% Copyright 2012-2016 The MathWorks, Inc.

  [S,R] = size(w);
  Q = size(p,2);
  if isa(w,'gpuArray')
    dz_dw = iGPU(p,R,S,Q);
  else
    dz_dw = iCPU(p,R,S,Q);
  end
end

function dz_dw = iCPU(p,R,S,Q)
  pt = reshape(p',1,Q,1,R); % 1xQx1xR
  dz_dw = zeros(S,Q,S,R); % SxQxSxR
  for i=1:S
    dz_dw(i,:,i,:) = pt;
  end
end

function dz_dw = iGPU(p,R,S,Q)
  pt = reshape(p',1,Q,1,R); % 1xQx1xR
  ind1 = gpuArray.colon(1,S)'; % Sx1x1x1
  ind3 = reshape(gpuArray.colon(1,S),1,1,S); % 1x1xSx1
  dz_dw = arrayfun(@iGPUHelper,pt,ind1,ind3);
end

function dz_dw = iGPUHelper(pt,ind1,ind3)

% If ind1==ind3, then dz_dw = pt, otherwise dz_dw = 0.
dz_dw = (ind1 == ind3)*pt;
end
