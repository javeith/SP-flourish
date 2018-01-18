classdef(Abstract) Reporter < handle
    methods(Abstract)
        start( this )
        reportIteration( this, epoch, iteration, elapsedTime, miniBatchLoss, miniBatchAccuracy, learnRate )
        reportEpoch( this, epoch, iteration, network )
        finish( this )
    end
end
