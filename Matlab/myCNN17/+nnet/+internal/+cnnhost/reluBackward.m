function dLossdX = reluBackward(Z, dLossdZ, X) %#ok<INUSL>
% reluBackward   Perform backpropagation for a ReLU layer
%
% Inputs:
% Z - The output from the ReLU layer. Unused, here to match cuDNN API.
% dLossdZ - The derivative of the loss function with respect to the output
% of the ReLU layer. A (H)x(W)x(C)x(N) array.
% X - The input to the ReLU layer. A (H)x(W)x(C)x(N) array.
%
% Output:
% dLossdX - The derivative of the loss function with respect to the input
% of the ReLU layer. A (H)x(W)x(C)x(N) array.

%   Copyright 2015-2016 The MathWorks, Inc.

dLossdX = dLossdZ .* (X > 0);
end
