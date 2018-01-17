function varargout = calculateJacobian(net,data,hints)
%calculateJacobian Jacobian values and performance

% Copyright 2015-2016 The MathWorks, Inc.
  
  % 1 or 3 Masks
  if (nargout <= 4)
    numMasks = 1;
  else
    numMasks = 3;
  end
  
  % Which Jacobian function should we use for the direction of computation?
  direction = iDirection( hints, net );
  jacobianFcn = iJacobianFunctionForDirection( direction );

  % Calculate gradient, with or without batching
  B = hints.batchSize;
  if isnan(B) || (B == data.Q)
    [je,jj,perfs,counts] = jacobianFcn(net,data,hints,numMasks);
  else
    [je,jj,perfs,counts] = iCalculateWithBatches(net,data,hints,jacobianFcn,numMasks);
  end
  
  % Cast to CPU double
  cpuDouble = double(1);
  je = cast(je,'like',cpuDouble);
  jj = cast(jj,'like',cpuDouble);
  perfs = cast(perfs,'like',cpuDouble);
  counts = cast(counts,'like',cpuDouble);
    
  % Assign output arguments
  varargout = [num2cell(perfs) {je jj} num2cell(counts)]; %#ok<VARARG>
end

function direction = iDirection(hints,net)
  direction = hints.direction;
  if strcmp(direction,'default')
      if (net.numLayerDelays == 0)
          direction = 'static';
      else
          direction = 'forward';
      end
  end
end

function jacobian = iJacobianFunctionForDirection( direction )
  switch direction
    case 'static'
      jacobian = @nnet.mode.matlab.calculateJacobianBackprop;
    case 'forward'
      jacobian = @nnet.mode.matlab.calculateJacobianForwardprop;
    case 'backward'
      jacobian = @nnet.mode.matlab.calculateJacobianBackprop;
  end
end

function [je,jj,perfs,counts] = iCalculateWithBatches(net,data,hints,jacobianFcn,numMasks)

  Q = data.Q; % All samples
  B = hints.batchSize; % Samples per batch
  
  % Allocate outputs
  je = zeros(net.numWeightElements,1,'like',hints.arrayType);
  jj = zeros(net.numWeightElements,net.numWeightElements,'like',hints.arrayType);
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
    [jeb,jjb,perfsb,countsb] = jacobianFcn(net,batch,hints,numMasks);

    % Accumulate results
    je = je + jeb;
    jj = jj + jjb;
    perfs = perfs + perfsb;
    counts = counts + countsb;
  end
end

