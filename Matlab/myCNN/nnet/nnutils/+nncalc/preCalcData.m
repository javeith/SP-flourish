function data = preCalcData(~,~,net,data,doPc,doPd,doFlattenTime)

% Copyright 2012-2016 The MathWorks, Inc.

if isa(data.X,'gpuArray')
    % Legacy support for NNDATA2GPU formatted data
    data = iPreCalcDataForNNDATA2GPU(net,data,doPc);
    
else
    % General case of cell-of-matrix data
    mode = nnet.mode.Matlab;
    hints = mode.netHints(net,mode.hints);
    
    % Process Inputs
    if doPc
        data.Pc = nnet.mode.matlab.processInputs(net,[data.Xi data.X],data.Q,hints);
        data.X = {};
        data.Xi = {};
    else
        data.Pc = {};
    end
    
    % Delay Inputs
    if doPd
        data.Pd = nnet.mode.matlab.delayInputs(net,data.Pc,data.Q,hints);
        data.Pc = {};
    else
        data.Pd = {};
    end
    
    % Flatten Time
    if doFlattenTime
        data = nncalc.flattenTime(data);
    end
end
end

function  data = iPreCalcDataForNNDATA2GPU(net,data,doPc)
precision = classUnderlying(data.X);
if doPc
    mode = nnGPU('precision',precision);
    hints = mode.netHints(net,mode.hints);
    data.Pc = mode.pc(net,data.X,data.Xi,data.Q,data.TS,hints);
    data.X = nndata2gpu([],precision);
    data.Xi = nndata2gpu([],precision);
else
    data.Pc = nndata2gpu([],precision);
end
data.Pd = nndata2gpu([],precision);
end
