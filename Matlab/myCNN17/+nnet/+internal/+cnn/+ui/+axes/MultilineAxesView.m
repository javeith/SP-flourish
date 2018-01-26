classdef MultilineAxesView < nnet.internal.cnn.ui.axes.AxesView
    % MultilineAxesView   View of an multiline axes
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % Panel   (uipanel) The parent panel of the axes
        Panel
    end
    
    properties(Access = private)
        % AxesModel   (nnet.internal.cnn.ui.axes.AxesModel)
        AxesModel
        
        % Axes   (axes) The main axes
        Axes
        
        % EpochRectangles   (patch) The patch object representing the epoch
        % rectangle backgrounds.
        EpochRectangles
        
        % EpochTexts   (array of text) The array of text objects that label
        % the epochs
        EpochTexts
        
        % EpochDisplayer   (nnet.internal.cnn.ui.axes.EpochDisplayer)
        % Helper class for drawing and updating the epochs
        EpochDisplayer
        
        % Lines   (cell of line) The lines added to the Axes
        Lines
    end
    
    methods
        function this = MultilineAxesView(axesModel, epochDisplayer, tagSuffix)
            this.Panel = uipanel('Parent', [], 'BorderType', 'none', 'Tag', 'NNET_CNN_TRAININGPLOT_AXESVIEW_PANEL');
            this.AxesModel = axesModel;
            this.EpochDisplayer = epochDisplayer;
            
            this.createGUIComponents(tagSuffix);
        end
        
        function update(this)
            % compute the bounds once only.
            xLim = this.AxesModel.XLim;
            yLim = this.AxesModel.YLim;
            
            % update everything 
            this.updateBounds(xLim, yLim);
            this.updateEpochs(xLim, yLim);
            this.updateLines();
        end
    end
    
    methods(Access = private)
        function createGUIComponents(this, tagSuffix)
            tag = sprintf('NNET_CNN_TRAININGPLOT_AXESVIEW_AXES_%s', tagSuffix);
            
            this.Axes = axes(...
                'Parent', this.Panel, ...
                'Tag', tag, ...
                'XGrid', 'off', ...
                'YGrid', 'on', ...
                'Position', [0.07, 0.15, 0.86, 0.82]);
            
            this.setLabels();
            
            xLim = this.AxesModel.XLim;
            yLim = this.AxesModel.YLim;
            
            this.updateBounds(xLim, yLim);
            
            this.ensureChildObjectsCanBeClipped();
            this.createEpochs();
            this.updateEpochs(xLim, yLim);
            
            this.createLines();
            this.updateLines();
            
            drawnow();
        end
        
        % lines
        function createLines(this)
            this.Lines = {};
            for i=1:numel(this.AxesModel.LineModels)
                lineModel = this.AxesModel.LineModels{i};
                tagSuffix = num2str(i);
                this.Lines{i} = iCreateLine(this.Axes, lineModel, tagSuffix);
            end
        end
        
        function updateLines(this)
            for i=1:numel(this.AxesModel.LineModels)
                lineModel = this.AxesModel.LineModels{i};
                this.Lines{i}.XData = lineModel.XValues;
                this.Lines{i}.YData = lineModel.YValues;
                this.Lines{i}.MarkerIndices = lineModel.MarkerIndices;
            end
        end
        
        % bounds
        function updateBounds(this, xLim, yLim)
            xlim(this.Axes, xLim);
            ylim(this.Axes, yLim);
        end
        
        % labels
        function setLabels(this)
            xlabel(this.Axes, this.AxesModel.XLabel, 'Interpreter', 'none');
            ylabel(this.Axes, this.AxesModel.YLabel, 'Interpreter', 'none');
        end
        
        % epochs
        function createEpochs(this)
            this.EpochRectangles = patch(this.Axes, 'XData', [], 'YData', [], 'Tag', 'NNET_CNN_TRAININGPLOT_AXESVIEW_EPOCHRECTANGLES');
            this.EpochTexts = matlab.graphics.primitive.Text.empty(0,1);
            for i=1:numel(this.AxesModel.EpochIndicesForTexts)
                this.EpochTexts(end+1) = text(this.Axes, 'String', '', 'Tag', 'NNET_CNN_TRAININGPLOT_AXESVIEW_EPOCHTEXTS'); 
            end
            this.EpochDisplayer.initializeEpochTexts(this.EpochTexts);
        end
        
        function updateEpochs(this, xLim, yLim)
            this.EpochDisplayer.updateEpochRectangles(...
                this.EpochRectangles, ...
                this.AxesModel.EpochInfo.NumEpochs, ...
                this.AxesModel.EpochInfo.NumItersPerEpoch, ...
                yLim);
            
            this.EpochDisplayer.updateEpochTexts(...
                this.EpochTexts, ...
                this.AxesModel.EpochIndicesForTexts, ...
                this.AxesModel.EpochInfo.NumItersPerEpoch, ...
                xLim, ...
                yLim);
        end
        
        % clipping
        function ensureChildObjectsCanBeClipped(this)
            % ensureChildObjectsCanBeClipped   Turning on 'Clipping'
            % ensures that each child object can control whether it gets
            % clipped by the Axes or not. Setting ClippingStyle to
            % 'rectangle' allows the part of the child that is within the
            % bounds of the axes to be shown, with the rest clipped off.
            this.Axes.Clipping = 'on';
            this.Axes.ClippingStyle = 'rectangle';
        end
    end
end

% helpers
function l = iCreateLine(parent, lineModel, tagSuffix)
tag = ['NNET_CNN_TRAININGPLOT_AXESVIEW_LINE_', tagSuffix];
l = line('Parent', parent, 'Tag', tag);
l.LineStyle  = lineModel.LineStyle;
l.LineWidth  = lineModel.LineWidth;
l.Color      = lineModel.LineColor;
l.Marker     = lineModel.MarkerType;
l.MarkerSize = lineModel.MarkerSize;
l.MarkerFaceColor = lineModel.MarkerFaceColor;
l.MarkerEdgeColor = lineModel.MarkerEdgeColor;

l.XData = lineModel.XValues;
l.YData = lineModel.YValues;
l.MarkerIndices = lineModel.MarkerIndices;
end

