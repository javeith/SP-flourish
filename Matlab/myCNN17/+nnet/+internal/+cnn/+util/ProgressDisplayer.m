classdef ProgressDisplayer < nnet.internal.cnn.util.Reporter
    % ProgressDisplayer   Class to display training progress
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties
        % Frequency   Frequency to report messages expressed in iterations.
        % This can be a scalar or a vector of frequencies
        Frequency = 50
    end
    
    properties(Access = private)
        % Iteration message to be reported
        IterationMessage = ''
        
        % Columns   (nnet.internal.cnn.util.ColumnStrategy) Column strategy
        Columns
    end
    
    methods
        function this = ProgressDisplayer( columnStrategy )
            this.Columns = columnStrategy;
        end
        
        function setup( ~ ) 
        end
        
        function start( this )
            iPrintString(this.Columns.HorizontalBorder);
            iPrintString(this.Columns.Headings);
            iPrintString(this.Columns.HorizontalBorder);
        end
        
        function reportIteration( this, summary )
            msg = this.buildMsgFromSummary( summary );
            this.storeIterationMessage( msg );
            if iCanPrint(summary.Iteration, this.Frequency)
                this.displayIterationMessage();
            end
        end
        
        function reportEpoch( ~, ~, ~, ~ )
        end
        
        function finish( this )
            this.displayIterationMessage();
            iPrintString(this.Columns.HorizontalBorder);
        end
    end
    
    methods(Access = private)
        function storeIterationMessage( this, msg )
            this.IterationMessage = msg;
        end
        
        function displayIterationMessage( this )
            % displayIterationMessage   Displays iteration message if it is
            % non-empty, then re-sets the iteration message to be empty.
            msg = this.IterationMessage;
            if ~isempty( msg )
                disp( msg );
            end
            this.IterationMessage = '';
        end
        
        function msg = buildMsgFromSummary( this, summary )
            names = this.Columns.Names;
            formats = this.Columns.Formats;
            formats = iDetermineLearnRateFormat( summary, formats );
            textPieces = iGetTextPiecesFromSummary( summary, names, formats );
            msg = strjoin( textPieces, iCentralDelimiter);
            msg = [iLeftDelimiter msg iRightDelimiter];
        end
    end
end

function iPrintString( str )
fprintf( '%s\n', str );
end

function tf = iCanPrint(iteration, frequency)
tf = any(mod(iteration, frequency) == 0) || (iteration == 1) ;
end

function delimiter = iCentralDelimiter()
delimiter = ' | ';
end

function delimiter = iLeftDelimiter()
delimiter = '| ';
end

function delimiter = iRightDelimiter()
delimiter = ' |';
end

function formats = iDetermineLearnRateFormat(summary, formats)
if isprop(summary, 'LearnRate')
    if summary.LearnRate < 1e-4
        formats{end} = '%12.2e';
    end
end
end

function textPieces = iGetTextPiecesFromSummary( summary, names, formats )
textPieces = cellfun(@(format, name)iGetTextFromSummary(summary, format, name), formats, names, 'UniformOutput', false );
end

function summaryText = iGetTextFromSummary(summary, format, name)
summaryValue = summary.(name);
if isempty(summaryValue)
    summaryText = iEmptyTextCell;
else
    summaryText = sprintf(format, summary.(name));
end
end

function textCell = iEmptyTextCell()
textCell = sprintf('%12s','');
end
