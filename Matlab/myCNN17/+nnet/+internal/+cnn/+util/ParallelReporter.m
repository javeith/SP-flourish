classdef ParallelReporter < nnet.internal.cnn.util.Reporter
    % ParallelReporter   Convolutional neural networks parallel training reporter interface
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties
        Reporter
        DataQueue
        HasCheckpointSavers
    end
    
    properties (Transient)
        % Transient so that it reverts to false when the object is copied
        % to the pool
        IsClient = false
    end
    
    methods
        function this = ParallelReporter( reporter )
            % ParallelReporter  Constructor - object must be created on client.
            this.Reporter = reporter;
            this.IsClient = true;
            this.HasCheckpointSavers = iDetectCheckpointSavers( reporter );
            this.DataQueue = parallel.internal.pool.DataQueue;
            
            % Set the function to be called when receiving data from the
            % pool
            this.DataQueue.afterEach( @this.report);
            
            % Forward interrupt events from the underlying reporter object
            addlistener( reporter, 'TrainingInterruptEvent', ...
                @(~,~)notify(this, 'TrainingInterruptEvent') );
        end
    end
    
    methods
        function report( this, args )
            % report - One-argument reporter for DataQueue to execute
            try
                feval( args{1}, this.Reporter, args{2:end} );
            catch exception
                % Errors in DataQueue continuations get ignored, so instead
                % return them in the TrainingInterruptEvent
                notify( this, 'TrainingInterruptEvent', ...
                    nnet.internal.cnn.util.TrainingInterruptEventData( exception ) );
            end
        end
        
        function setup( this )
            reportOrSendToClient( this, 'setup' );
        end
        
        function start( this )
            reportOrSendToClient( this, 'start' );
        end
        
        function reportIteration( this, varargin )
            reportOrSendToClient( this, 'reportIteration', varargin{:} );
        end
        
        function computeIteration( this, varargin )
            % This method will not send data to client but will execute
            % locally on the first worker to avoid multiple reports of the
            % same
            if labindex == 1
                this.Reporter.computeIteration( varargin{:} );
            end
        end
        
        function reportEpoch( this, varargin )
            % If there are no CheckpointSavers then there is no need to
            % include the network parameter. Deleting it avoids unnecessary
            % data transfer to the client.
            if numel(varargin) >= 3 && ~this.HasCheckpointSavers
                % The parameter needs to be there to satisfy number of
                % arguments, but it can be anything, so make empty
                varargin{3} = [];
            end
            reportOrSendToClient( this, 'reportEpoch', varargin{:} );
        end
        
        function finish( this )
            reportOrSendToClient( this, 'finish' );
        end
    end
    
    methods (Access = private)
        function reportOrSendToClient( this, method, varargin )
            if this.IsClient
                % If on client just report as normal
                feval( method, this.Reporter, varargin{:} );
            else
                % If on worker, report, but only from worker 1 to avoid
                % multiple reports
                if labindex == 1
                    % Send to client for the client to report
                    send( this.DataQueue, { method, varargin{:} } ); %#ok<CCAT>
                end
            end
        end
    end
end

function iReportProgressDisplayersOnly( reporter, method, varargin )
if isa( reporter, 'nnet.internal.cnn.util.VectorReporter' )
    for i = 1:length( reporter.Reporters )
        iReportProgressDisplayersOnly( reporter.Reporters{i}, method, varargin{:} );
    end
elseif isa( reporter, 'nnet.internal.cnn.util.ProgressDisplayer' )
    feval( method, reporter, varargin{:} );
end
end

function hasCheckpointing = iDetectCheckpointSavers( reporter )
% Look for any CheckpointSaver reporters
hasCheckpointing = false;
if isa( reporter, 'nnet.internal.cnn.util.VectorReporter' )
    for i = 1:length( reporter.Reporters )
        hasCheckpointing = iDetectCheckpointSavers( reporter.Reporters{i} );
        if hasCheckpointing
            return;
        end
    end
elseif isa( reporter, 'nnet.internal.cnn.util.CheckpointSaver' )
    hasCheckpointing = true;
end
end
