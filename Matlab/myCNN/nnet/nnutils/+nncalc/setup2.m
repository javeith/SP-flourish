function [calcLib,calcNet] = setup2(calcMode,calcNet,calcData,calcHints,isParallel)
% Second step of setup for calculation mode, net, data & hints for
% parallel or non-parallel calculations.  Call this function from inside
% a SPMD block for parallel calculations, outside for non-parallel.

% Copyright 2012-2015 The MathWorks, Inc.

% Net and Data, formatting and hints
if calcHints.isActiveWorker
  
  if strcmp(calcMode.mode,'nnet.mode.Matlab')
    % IMPROVED/SIMPLIFIED SETUP
    % TO REPLACE CODE BELOW IN CALC MODE CLEANUP TASK
    % Change 1, dataHints called before formatNet, does not take calcNet
    % Change 2, formatData takes calcNet
    calcHints = calcMode.dataHints(calcData,calcHints);
    calcNet = calcMode.formatNet(calcNet,calcHints);
    calcData = calcMode.formatData(calcNet,calcData,calcHints);
    calcHints = calcMode.codeHints(calcHints);
    
  else
    
    calcNet = calcMode.formatNet(calcNet,calcHints);
    calcHints = calcMode.dataHints(calcNet,calcData,calcHints);
    calcData = calcMode.formatData(calcData,calcHints);
    calcHints = calcMode.codeHints(calcHints);
  end
else
  calcNet = {};
  calcData = {};
end

% Wrap calcMode in main
calcLib = nnCalcLib(calcMode,calcData,calcHints);
