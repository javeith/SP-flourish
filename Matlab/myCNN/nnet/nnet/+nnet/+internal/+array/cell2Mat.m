function y = cell2Mat(x)
%CELL2MAT Version of CELL2MAT that works with gpuArrays

% Copyright 2015 The MathWorks, Inc.

if isempty(x)
  y = [];
else
  [rows,cols] = size(x);
  rowSizes = cellfun(@(x) size(x,1),x(:,1)');
  colSizes = cellfun(@(x) size(x,2),x(1,:));
  y = zeros(sum(rowSizes),sum(colSizes),'like',x{1});
  rowPos = cumsum([0 rowSizes]);
  colPos = cumsum([0 colSizes]);
  for i=1:rows
    rowInd = (rowPos(i)+1):rowPos(i+1);
    for j=1:cols
      colInd = (colPos(j)+1):colPos(j+1);
      y(rowInd,colInd) = x{i,j};
    end
  end
end
