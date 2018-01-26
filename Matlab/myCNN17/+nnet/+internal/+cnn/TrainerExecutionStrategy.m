classdef TrainerExecutionStrategy
   % TrainerExecutionStrategy   Interface for Trainer execution strategies
   %
   %   A class that inherits from this interface will be used to implement
   %   calculations in the internal Trainer class on either the host or
   %   GPU.
   
   % Copyright 2016 The Mathworks, Inc.
   
    methods (Abstract)
       
        Y = environment(this, X)
        % Y = environment(this, X) is used by the Trainer to ensure that
        % data X used for training correctly associated with the hardware
        % used for training.
        
        [accumI, numImages] = computeAccumImage(this, distributedData, augmentations );
        % [accumI, numImages] = computeAccumImage(this, distributedData,
        % augmentations ) computes the accumulated image of the training
        % data, which is used during the average image calculation.
        
    end
    
    methods (Access = public)
        
        function avgI = computeAverageImage(this, data, augmentations, executionSettings)
            % Average image is computed in parallel or in serial
            if executionSettings.useParallel
                avgI = this.computeAverageImageParallel(data, augmentations);
            else
                avgI = this.computeAverageImageSerial(data, augmentations);
            end
            avgI = gather( avgI );
        end
       
        function avgI = computeAverageImageSerial(this, data, augmentations)
            [accumI, numImages] = this.computeAccumImage(data, augmentations);
            avgI = accumI ./ numImages;
        end
        
        function avgI = computeAverageImageParallel(this, data, augmentations)
            % Accumulate
            distributedData = data.DistributedData;
            spmd
                % Every worker needs to know which lab has the result
                hasData = distributedData.NumObservations > 0;
                labIndexWithData = labindex;
                if ~hasData
                    labIndexWithData = inf;
                end
                labIndexWithResults = gop(@min, labIndexWithData);
                
                % Split the communicator so that only workers contributing data have to
                % communicate
                comm = feval('distributedutil.CommSplitter', double(hasData)+1, labindex);
                if hasData
                    % Compute average on each worker
                    [accumI, numImages] = this.computeAccumImage( distributedData, augmentations );
                    % Compute combined average
                    accumI = gplus(accumI, 1, class(accumI));
                    numImages = gplus(numImages, 1);
                else
                    accumI = [];
                    numImages = 0;
                end
                delete(comm);
                
                % Always output on the host in case client has no GPU to copy to
                accumI = gather( accumI );
                
                % Return results as distributedutil.AutoDeref objects that therefore do
                % not need an extra remote call to gather the contents back to the
                % client
                accumI = feval('distributedutil.AutoTransfer', accumI, labIndexWithResults );
                numImages = feval('distributedutil.AutoTransfer', numImages, labIndexWithResults );
            end
            % Gather back to the client
            avgI = accumI.Value ./ numImages.Value;
        end
                
    end
    
end