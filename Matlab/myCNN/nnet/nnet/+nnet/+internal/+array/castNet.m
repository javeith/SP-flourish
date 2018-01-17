function net = castNet(net,arrayType)
 
% Copyright 2015 The MathWorks, Inc.

  % Cast weights and biases to match mode
  net = nnet.internal.array.castAll(net,arrayType,{'b','IW','LW'});
