function data = formatData(data1,hints)

  % Copyright 2013-2016 The MathWorks, Inc.

  % Simulation Data
  data.X = iEnsureFull(double(cell2mat(data1.X)));
  data.Xi = iEnsureFull(double(cell2mat(data1.Xi)));
  data.Pc = iEnsureFull(double(cell2mat(data1.Pc)));
  data.Pd = iEnsureFull(double(iFormatPd(data1.Pd)));
  data.Ai = iEnsureFull(double(cell2mat(data1.Ai)));

  data.Q = data1.Q;
  data.TS = data1.TS;

  % Performance Data
  if isfield(data1,'T')
    data.T = iEnsureFull(double(cell2mat(data1.T)));
    data.EW = iEnsureFull(double(cell2mat(data1.EW)));
    if isfield (data1,'train')
      data.masks = double(cell2mat([data1.train.mask data1.val.mask data1.test.mask]));
      data.trainMask = double(cell2mat(data1.train.mask));
    end
  end
end

function Pd = iFormatPd(Pd)
  if isempty(Pd)
    Pd = [];
  else
    matrix2vector = @(x) x(:);
    Pd = cellfun(matrix2vector,Pd,'UniformOutput',false);
    Pd = cat(1,Pd{:});
  end
end

function X = iEnsureFull( X )
if issparse( X )
    warning( message( 'nnet:NNData:ConvertSparseToFull' ) );
    X = full( X );
end
end
