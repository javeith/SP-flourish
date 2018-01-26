function [returnSeq, statefulLayers] = returnsSequence( internalLayers, isRNN )
% returnsSequence   Determine if an RNN layer array is configured such that
% its output is a sequence, or a single element.

%   Copyright 2017 The MathWorks, Inc.

returnSeq = false;
statefulLayers = false( numel(internalLayers), 1 );
if isRNN
    rnnLayers = cellfun(@(x) isa(x,'nnet.internal.cnn.layer.Updatable'), internalLayers);
    layerIndices = find( rnnLayers );
    statefulLayers( rnnLayers ) = true;
    numRNNLayers = numel(layerIndices);
    layerReturnSeq = false( numRNNLayers, 1 );
    for ii = 1:numRNNLayers
        layerReturnSeq(ii) = internalLayers{ layerIndices(ii) }.ReturnSequence;
    end
    returnSeq = all( layerReturnSeq );
end
end