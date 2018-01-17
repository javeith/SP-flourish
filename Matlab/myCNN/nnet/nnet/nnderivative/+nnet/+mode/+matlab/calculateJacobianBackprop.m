function [je,jj,perfs,counts] = calculateJacobianBackprop(net,data,hints,numMasks)
%CALCULATEJACOBIANBACKPROP Calculate Jacobian values with backpropagation

% Copyright 2015-2016 The MathWorks, Inc.

  import nnet.mode.matlab.getProcessedInputStates
  import nnet.mode.matlab.getProcessedInputs
  import nnet.mode.matlab.evaluateLayer
  import nnet.mode.matlab.processOutput
  import nnet.mode.matlab.applyObservationWeights
  import nnet.mode.matlab.applyMasks
  import nnet.mode.matlab.backpropLayerJacobian
  import nnet.mode.matlab.wrapTimesteps
  
  % Allocate outputs
  je = zeros(net.numWeightElements,1,'like',hints.arrayType);
  jj = zeros(net.numWeightElements,net.numWeightElements,'like',hints.arrayType);
  perfs = zeros(1,numMasks,'like',hints.arrayType);
  counts = zeros(1,numMasks,'like',hints.arrayType);
  
  % For backprop gradient the full time windows are needed
  inputTimeWindow = net.numInputDelays + data.TS;
  layerTimeWindow = net.numLayerDelays + data.TS;

  % Allocate Temporary Values
  Ac = [data.Ai cell(net.numLayers,data.TS)];
  N = cell(net.numLayers,data.TS);
  Z = cell(net.numLayers,hints.maxZ,data.TS);
  
  % Preprocess Initial Input States
  Xp = getProcessedInputStates(net,data,hints,inputTimeWindow);

  % Evaluate Forward in Time
  for ts = 1:data.TS

    % Preprocess Inputs
    Xp = getProcessedInputs(net,data,Xp,ts,hints,inputTimeWindow);

    % Layers
    for i = hints.layerOrder

      timeslot = net.numLayerDelays + ts;
      [a,n,z] = evaluateLayer(net,i,ts,data,Xp,Ac,hints,inputTimeWindow,layerTimeWindow);
      [Ac{i,timeslot},N{i,ts},Z(i,1:hints.numZ(i),ts)] = deal(a,n,z);
      
      % Outputs
      if net.outputConnect(i)
        
        % Output Post-processing
        ii = hints.layer2Output(i);
        [y,Yp] = processOutput(Ac{i,timeslot},ii,hints);
        
        % Error
        t = data.T{ii,ts};
        e = t - y;
    
        % Error Normalization
        if hints.doErrNorm(ii)
          e = bsxfun(@times,e,hints.errNorm{ii});
        end
        
        % Performance
        perf = hints.perfApply(t,y,e,hints.perfParam);
        
        % Observation Weights
        if hints.doEW
          [perf,ew] = applyObservationWeights(perf,data.EW,ii,ts,hints);
          sqrtew = sqrt(ew);
          e = bsxfun(@times,e,sqrtew);
        end

        % Masks
        [perfs,counts,indOfNaN] = applyMasks(perfs,counts,ii,ts,perf,data.MASKS,numMasks);
        
        % Derivative of Outputs with respect to Errors
        S = net.outputs{i}.size;
        dy = -ones(net.outputs{i}.size,data.Q);
        
        if hints.doErrNorm(ii)
          dy = bsxfun(@times,dy,hints.errNorm{ii});
        end
        if hints.doEW
          dy = bsxfun(@times,dy,sqrtew);
        end
        dy(indOfNaN) = 0;
        e(indOfNaN) = 0;
        
        % Backprop from output, 1 element at a time
        for k = 1:S
          
          % Initialize derivates to 0 or empty
          [dB,dIW,dLW,dA] = iInitializeDerivatives(net,data.Q,data.arrayType,hints);
          
          % Backpropagate kth row of output derivatives
          dy1 = zeros(S,data.Q,'like',data.arrayType);
          dy1(k,:) = dy(k,:);
        
          % Backprop through Output Processing
          for j = 1:hints.numOutProc(ii)
            dy1 = hints.out(ii).procBPrev{j}(dy1,Yp{j},Yp{j+1},hints.out(ii).procSet{j});
          end
          %timestep = 
          dA{i,ts} = dy1;

          % Backprop through time
          for ts_bp = ts:-1:1

            % Backprop through layers
            for i_bp = hints.layerOrderReverse
              if ~isempty(dA{i_bp,ts_bp})
                [dA,dB,dIW,dLW] = ...
                  backpropLayerJacobian(dA,dB,dIW,dLW,net,i_bp,ts_bp,data,Xp,Z,N,Ac,hints);
              end
            end
          end
          
          [jj,je] = iAccumulateJEJJ(jj,je,dB,dIW,dLW,e(k,:),data.Q,hints);
          
        end % k - output element
      end % output
    end % i - layer
  end % ts - timestep
end

function [dB,dIW,dLW,dA] = iInitializeDerivatives(net,Q,arrayType,hints)
  dB = cell(net.numLayers,1);
  dIW = cell(net.numLayers,net.numInputs);
  dLW = cell(net.numLayers,net.numLayers);
  for i=1:net.numLayers
    if hints.bInclude(i)
      dB{i} = zeros(net.layers{i}.size,Q,'like',arrayType);
    end
    for j=1:net.numInputs
      if hints.iwInclude(i,j)
        dIW{i,j} = zeros([net.inputWeights{i,j}.size,Q],'like',arrayType);
      end
    end
    for j=1:net.numLayers
      if hints.lwInclude(i,j)
        dLW{i,j} = zeros([net.layerWeights{i,j}.size,Q],'like',arrayType);
      end
    end
  end
  dA = cell(net.numLayers,net.numLayerDelays+1);
end

function [jj,je] = iAccumulateJEJJ(jj,je,dB,dIW,dLW,e,Q,hints)
  dIW = cellfun(@(w) iReshapeWeightDerivatives(w,Q),dIW,'UniformOutput',false);
  dLW = cellfun(@(w) iReshapeWeightDerivatives(w,Q),dLW,'UniformOutput',false);
  
  jac = [dB dIW dLW]';
  jac = cat(1,jac{:});
  
  % Jacobian * error and Jacobian' * Jacobian
  je = je + jac * e(:);
  jj = jj + jac * jac';
end

function w = iReshapeWeightDerivatives(w,Q)
  if ~isempty(w)
    w = reshape(w,numel(w)/Q,Q);
  end
end
