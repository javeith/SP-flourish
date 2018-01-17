function [userStop,userCancel] = stop_or_cancel()
%stop_or_cancel   Check if user has pressed stop or cancel. In deployed mode
% we can't rely on the presence of nntraintool when checking for user stop
% or cancel. Therefore we just assume no stop or cancel when deployed.
if isdeployed
    userStop = false;
    userCancel = false;
else
    [userStop,userCancel] =  nntraintool('check');
end
end