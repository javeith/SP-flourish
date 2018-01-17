function net = formatNet(net,hints)

% Copyright 2012-2015 The MathWorks, Inc.

% Cast weights and biases to match mode
net = nnet.internal.array.castNet(net,hints.arrayType);
