function varargout = calculateGradient(net,data,hints)
%CALCULATEGRADIENT Derivative of performance with respect to weights

% Copyright 2015-2016 The MathWorks, Inc.

% 1 or 3 Masks
if (nargout <= 3)
    numMasks = 1;
else
    numMasks = 3;
end

% Forward or backward gradient
gradientFcn = iChooseGradientFunction(hints.direction);

% Calculate gradient, with or without batching
B = hints.batchSize;
if isnan(B) || (B == data.Q)
    [grad,perfs,counts] = gradientFcn(net,data,hints,numMasks);
else
    [grad,perfs,counts] = iCalculateWithBatches(net,data,hints,gradientFcn,numMasks);
end

% Cast to CPU double
cpuDouble = double(1);
grad = cast(grad,'like',cpuDouble);
perfs = cast(perfs,'like',cpuDouble);
counts = cast(counts,'like',cpuDouble);

% Output arguments
%
% The two output different argument formats implement the interfaces
% expected by nnCalcLib.grad (3 outputs) and nnCalcLib.perfsGrad (7
% arguments). The inconsistent ordering is an artifact of other calculation
% modes such as nnMex, which implement these API’s with different header
% functions instead of a single function.
if (nargout <= 3)
    % Calculates training gradient, performance and counts
    varargout = {grad perfs counts};
else
    % Calculates training, validation and test performance,
    %   the training gradient, and training, validation and test counts.
    varargout = [num2cell(perfs) {grad} num2cell(counts)]; %#ok<VARARG>
end

end

function gradientFcn = iChooseGradientFunction(direction)
switch direction
    case {'default','backward'}
        gradientFcn = @nnet.mode.matlab.calculateGradientBackprop;
    case 'forward'
        gradientFcn = @nnet.mode.matlab.calculateGradientForwardprop;
end
end

function [grad,perfs,counts] = iCalculateWithBatches(net,data,hints,gradientFcn,numMasks)

Q = data.Q; % All samples
B = hints.batchSize; % Samples per batch

% Allocate outputs
grad = zeros(hints.wbLen,1,'like',hints.arrayType);
perfs = zeros(1,numMasks,'like',hints.arrayType);
counts = zeros(1,numMasks,'like',hints.arrayType);

% Iterate over batches
for batchStart = 1:B:Q
    
    % Get batch of data
    % (Each batch is size B, except last batch may be less)
    batchStop = min(batchStart + B - 1,Q);
    batchIndices = batchStart:batchStop;
    batchSize = batchStop - batchStart + 1;
    batch = nnet.internal.data.getBatch(data,Q,batchIndices,...
        {'X','Xi','Xp','Xd','Ai','T','EW','MASKS'});
    batch.Q = batchSize;
    
    % Calculate gradient
    [gradb,perfsb,countsb] = gradientFcn(net,batch,hints,numMasks);
    
    % Accumulate results
    grad = grad + gradb;
    perfs = perfs + perfsb;
    counts = counts + countsb;
end
end
