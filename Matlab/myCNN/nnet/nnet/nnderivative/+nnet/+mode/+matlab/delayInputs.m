function Xd = delayInputs(net,Xp,Q,hints)
%DELAYINPUTS Apply delays to inputs

% Copyright 2015-2016 The MathWorks, Inc.

TS = size(Xp,2) - net.numInputDelays;
Xd = cell(net.numLayers,net.numInputs,TS);

for ts=1:TS
    for i=1:net.numLayers
        for j = 1:net.numInputs
            if net.inputConnect(i,j)
                Xd{i,j,ts} = nnet.mode.matlab.delayInput(net,i,j,ts,Xp,hints);
            else
                Xd{i,j,ts} = zeros(0,Q);
            end
        end
    end
end
end
