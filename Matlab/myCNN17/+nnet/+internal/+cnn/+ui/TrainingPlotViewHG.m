classdef TrainingPlotViewHG < nnet.internal.cnn.ui.TrainingPlotView
    % TrainingPlotViewHG   View for the training plot (HG).
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % Figure   (figure) The figure for the view.
        Figure
    end
    
    properties
        % ProgressPanelVisible   (logical) Is the panel showing all the
        % progress information visible?
        ProgressPanelVisible = true 
        
        % TrainingErrorMessageVisible    (logical) Is the message saying
        % that training errored out visible?
        TrainingErrorMessageVisible = false
        
        % PlotErrorMessageVisible    (logical) Is the message saying that
        % the plot errored out visible?
        PlotErrorMessageVisible = false
    end
    
    properties(Dependent, SetAccess = private)
        % FigureExistsAndIsVisible   (logical) Is the main figure visible
        % and existent?
        FigureExistsAndIsVisible 
    end
    
    properties(Access = private)
        % MainPanel   (uipanel) The panel containing everything.
        MainPanel
        
        % PlotTitle   (uicontrol) The uicontrol that shows the plot title
        PlotTitle
        
        % TopPlotPanel   (uipanel) The panel containing the top axes
        TopPlotPanel
        
        % BottomPlotPanel   (uipanel) The panel containing the bottom axes
        BottomPlotPanel
        
        % InfoStrip   (uipanel) The infostrip on the right hand side
        InfoStrip
        
        % InfoStripFlexibleSpacer  (uipanel) The empty panel in the infostrip
        %which can resize.
        InfoStripFlexibleSpacer
        
        % TrainingErrorMessageText   (uicontrol) Text in infostrip
        % explaining that an error occurred during training
        TrainingErrorMessageText
        
        % PlotErrorMessageText   (uicontrol) Text in infostrip explaining
        % that an error occurred in the plot
        PlotErrorMessageText
        
        % ProgressPanel   (uipanel) The panel containing all the progress
        % information
        ProgressPanel
        
        % DeterminateProgress   (nnet.internal.cnn.ui.progress.DeterminateProgress)
        % The progress bar in the infostrip.
        DeterminateProgress
        
        % DeterminateProgressListener   (listener) Listens for stop button
        % click
        DeterminateProgressListener
        
        % TextTablePanel   (uipanel) The panel containing the text table
        TextTablePanel
        
        % TextLayout   (nnet.internal.cnn.ui.layout.TextLayout)
        % The text table in the infostrip.
        TextLayout
        
        % LegendPanel   (uipanel) The panel containing the legend
        LegendPanel
        
        % LegendLayout   (nnet.internal.cnn.ui.layout.LegendLayout)
        % The Legend in the infostrip.
        LegendLayout
        
        % UserData   (anything) Arbitrary user data
        UserData
    end
    
    methods
        function this = TrainingPlotViewHG(determinateProgress, legendLayout, textLayout)
            this.DeterminateProgress = determinateProgress;
            this.LegendLayout = legendLayout;
            this.TextLayout = textLayout;
            this.createGUIComponents(); 
        end
        
        function delete(this)
            if ~isempty(this.Figure) && isvalid(this.Figure)
                delete(this.Figure);
            end
        end
        
        function closeFigure(this)
            delete(this.Figure); 
        end
        
        function setUserData(this, userData)
            this.UserData = userData; 
        end
        
        function showMainPlot(this)
            this.Figure.Visible = 'on';
            drawnow();
        end
        
        function setTitleInfo(this, trainingStartTime)
            startTimeStr = iDateTimeAsStringUsingDefaultLocalFormat(trainingStartTime);
            figureTitle = message('nnet_cnn:internal:cnn:ui:trainingplot:FigureTitle', startTimeStr).getString();
            
            this.Figure.Name = figureTitle;
            this.PlotTitle.String = figureTitle;
        end
        
        function setupProgress(this, currValue, maxValue)
            this.DeterminateProgress.Maximum = maxValue;
            this.DeterminateProgress.Value = currValue;
        end
        
        function updateProgress(this, currValue)
            this.DeterminateProgress.Value = currValue; 
        end
        
        function setupLegend(this, cellOfLegendSectionNames, cellOfLegendSectionStructArrs, checkboxesVisible)
            assert(numel(cellOfLegendSectionNames) == numel(cellOfLegendSectionStructArrs));
            for i=1:numel(cellOfLegendSectionNames)
               this.LegendLayout.addSection(cellOfLegendSectionNames{i}, cellOfLegendSectionStructArrs{i}); 
            end
            this.LegendLayout.AreCheckboxesVisible = checkboxesVisible;
            
            % ensure that legend has size as preferred size
            preferredHeightInPixels = this.LegendLayout.PreferredHeight / iGetPointsPerPixel();
            preferredWidthInPixels = this.LegendLayout.PreferredWidth / iGetPointsPerPixel();
            iSetPreferredHeight(this.LegendPanel, preferredHeightInPixels);
            iSetPreferredWidth(this.LegendPanel, preferredWidthInPixels);
            
            this.redrawInfoStrip();
        end
        
        function setupTableOfData(this, cellOfTableSectionNames, cellOfTableSectionStructArrs)
            assert(numel(cellOfTableSectionNames) == numel(cellOfTableSectionStructArrs));
            
            this.TextLayout.reset();
            for i=1:numel(cellOfTableSectionNames)
                this.TextLayout.addSection(cellOfTableSectionNames{i}, cellOfTableSectionStructArrs{i}); 
            end
            
            % ensure that textlayout has correct height
            preferredHeightInPixels = this.TextLayout.PreferredHeight / iGetPointsPerPixel();
            iSetPreferredHeight(this.TextTablePanel, preferredHeightInPixels);
        end
        
        function updateTableOfData(this, rowID, newRightText)
            this.TextLayout.update(rowID, newRightText);
        end
        
        function setTopAxes(this, axesView)
            axesView.Panel.Parent = this.TopPlotPanel;
        end
        
        function setBottomAxes(this, axesView)
            axesView.Panel.Parent = this.BottomPlotPanel;
        end

        function set.ProgressPanelVisible(this, tf)
            this.ProgressPanelVisible = tf;
            this.ProgressPanel.Visible = iBooleanToStr(tf); %#ok<MCSUP>
        end
        
        function set.TrainingErrorMessageVisible(this, tf)
            this.TrainingErrorMessageVisible = tf;
            this.TrainingErrorMessageText.Visible = iBooleanToStr(tf); %#ok<MCSUP>
            drawnow();
        end
        
        function set.PlotErrorMessageVisible(this, tf)
            this.PlotErrorMessageVisible = tf;
            this.PlotErrorMessageText.Visible = iBooleanToStr(tf); %#ok<MCSUP>
            drawnow();
        end
        
        function tf = get.FigureExistsAndIsVisible(this)
            tf = ~isempty(this.Figure) && isvalid(this.Figure) && strcmp(this.Figure.Visible, 'on');
        end
    end
    
    methods(Access = private)
        function createGUIComponents(this)
            this.createFigure();
            this.createMainPanel(this.Figure);
            drawnow();
        end
        
        function createFigure(this)
            this.Figure = nnet.internal.cnn.ui.figure(...
                'Tag', 'NNET_CNN_TRAININGPLOT_FIGURE', ...
                'Visible', 'off', ...
                'CloseRequestFcn', @this.closeRequestFcnCallback);
            this.Figure.UserData = this;  % Ensure that this View is persistent as long as the figure is.
            iResizeAndCenterFigure(this.Figure);
        end
        
        function createMainPanel(this, parent)
            this.MainPanel = uipanel(parent, 'BorderType', 'none');
            horizontalFlow = uiflowcontainer('v0', 'Parent', this.MainPanel, 'FlowDirection', 'lefttoright');
            this.createPlotPanel(horizontalFlow);
            this.createInfoStrip(horizontalFlow);
        end
        
        function createPlotPanel(this, parent)
            panel = uipanel('Parent', parent, 'BorderType', 'none'); 
            verticalFlow = uiflowcontainer('v0', 'Parent', panel, 'FlowDirection', 'topdown');
            
            % Plot Title
            this.PlotTitle = uicontrol('Parent', verticalFlow, 'Style', 'text', 'String', '', 'FontSize', 13, 'FontWeight', 'bold', 'Tag', 'NNET_CNN_TRAININGPLOT_PLOTTITLE');
            iSetPreferredHeight(this.PlotTitle, 30);
            
            % Plot Panels
            verticalLayout = nnet.internal.cnn.ui.layout.VerticalLayout(verticalFlow);
            %
            % create individual panels for the plots.
            this.TopPlotPanel    = uipanel('Parent', [], 'BorderType', 'none', 'Tag', 'NNET_CNN_TRAININGPLOT_TOPPLOTPANEL');
            this.BottomPlotPanel = uipanel('Parent', [], 'BorderType', 'none', 'Tag', 'NNET_CNN_TRAININGPLOT_BOTTOMPLOTPANEL');
            %
            % add the panels to the layout
            verticalLayout.add(this.TopPlotPanel,    0.50);
            verticalLayout.add(this.BottomPlotPanel, 0.30);
        end
        
        function createInfoStrip(this, parent)
            this.InfoStrip = uipanel('Parent', parent);
            iSetPreferredWidth(this.InfoStrip, 315);
            
            verticalFlow = uiflowcontainer('v0', 'Parent', this.InfoStrip, 'FlowDirection', 'topdown');
            
            iAddFixedHeightSpacer(verticalFlow, 5);
            this.addErrorMessages(verticalFlow);
            this.addProgressPanel(verticalFlow);
            iAddFixedHeightSpacer(verticalFlow, 15);
            this.addTextTable(verticalFlow);
            this.addHelpLink(verticalFlow);
            this.InfoStripFlexibleSpacer = iAddFlexibleSpacer(verticalFlow);
            this.addLegend(verticalFlow);
        end
        
        function addErrorMessages(this, parent)
            this.TrainingErrorMessageText = uicontrol(...
                'Parent', parent, ...
                'Style', 'text', ...
                'String', iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripTrainingErrorMessage'), ...
                'ForegroundColor', 'red', ...
                'FontWeight', 'bold', ...
                'HorizontalAlignment', 'left', ...
                'Visible', iBooleanToStr(this.TrainingErrorMessageVisible), ...
                'Tag', 'NNET_CNN_TRAININGPLOT_TRAININGERRORMESSAGE');
            iSetPreferredHeight(this.TrainingErrorMessageText, 50);
            
            this.PlotErrorMessageText = uicontrol(...
                'Parent', parent, ...
                'Style', 'text', ...
                'String', iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripPlotErrorMessage'), ...
                'ForegroundColor', 'red', ...
                'FontWeight', 'bold', ...
                'HorizontalAlignment', 'left', ...
                'Visible', iBooleanToStr(this.PlotErrorMessageVisible), ...
                'Tag', 'NNET_CNN_TRAININGPLOT_PLOTERRORMESSAGE');
            iSetPreferredHeight(this.PlotErrorMessageText, 50);
        end
        
        function addProgressPanel(this, parent)
            this.ProgressPanel = uipanel(...
                'Parent', parent, ...
                'BorderType', 'none', ...
                'Visible', iBooleanToStr(this.ProgressPanelVisible), ...
                'Tag', 'NNET_CNN_TRAININGPLOT_PROGRESSPANEL');
            
            horizontalFlow = uiflowcontainer('v0', ...
                'Parent', this.ProgressPanel, ...
                'FlowDirection', 'lefttoright', ...
                'Margin', iTinyMargin(), ...
                'Tag', 'NNET_CNN_TRAININGPLOT_DETERMINATEPROGRESSPARENT');
            
            % Add DeterminateProgress to row
            this.DeterminateProgress.Parent = horizontalFlow;
            iAddFixedWidthSpacer(horizontalFlow, 5);
            this.DeterminateProgressListener = addlistener(this.DeterminateProgress, 'StopButtonClicked', @this.stopButtonClickedCallback);
            
            % Set height of ProgressPanel now that it has been populated
            iSetPreferredHeight(this.ProgressPanel, this.DeterminateProgress.PreferredHeight);
        end
        
        function addTextTable(this, parent)
            this.TextTablePanel = uipanel('Parent', parent, 'BorderType', 'none', 'Tag', 'NNET_CNN_TRAININGPLOT_TEXTTABLEPARENT');
            this.TextLayout.Parent = this.TextTablePanel;
        end        
            
        function addHelpLink(this, parent)
            learnMoreLinkText = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripLearnMoreHelpLink');
            h = uicontrol(...
                'Parent', parent, ...
                'Style', 'text', ...
                'String', learnMoreLinkText, ...
                'FontWeight', 'bold', ...
                'Enable', 'Inactive', ...
                'HorizontalAlignment', 'left', ...
                'ForegroundColor', nnet.internal.cnn.ui.factory.Colors().Hyperlink, ...
                'ButtonDownFcn', @this.helpLinkClickedCallback, ...
                'Tag', 'NNET_CNN_TRAININGPLOT_HELPLINK');
            iSetPreferredHeight(h, 30);
            
            parentFigure = iFindFigure(parent);
            parentFigure.WindowButtonMotionFcn = @(fig,callbackdata) iChangeCursorOnMouseOverObject(h, fig, callbackdata);
        end
        
        function addLegend(this, parent)
            this.LegendPanel = uipanel('Parent', parent, 'BorderType', 'none', 'Tag', 'NNET_CNN_TRAININGPLOT_LEGENDPARENT');
            this.LegendLayout.Parent = this.LegendPanel;
        end
        
        function addJavaActionListener(this, javaObject, callback)
            h = handle(javaObject, 'callbackproperties');
            this.StopButtonListener = handle.listener(h, 'ActionPerformed', callback);
        end
        
        function redrawInfoStrip(this)
            this.redrawLegend();
        end
        
        function redrawLegend(this)
            % Prevent issues with resizing the GUI window (making the window
            % smaller causes the legend to become clipped, as
            % InfoStripFlexibleSpacer does not shrink preferentially as
            % expected; instead LegendPanel and InfoStripFlexibleSpacer
            % shrink at the same rate, which is not intended.

            this.LegendPanel.HeightLimits = [1 Inf];
            this.InfoStripFlexibleSpacer.HeightLimits = [1 1];
        end

        % callbacks
        function helpLinkClickedCallback(this, ~, ~)
            this.notify('HelpLinkClicked'); 
        end
        
        function stopButtonClickedCallback(this, ~, ~)
            this.notify('StopButtonClicked'); 
        end
        
        function closeRequestFcnCallback(this, ~, ~)
            this.notify('FigureCloseRequested'); 
        end
    end
end

% helpers
function str = iMessageString(varargin)
m = message(varargin{:});
str = m.getString();
end

function str = iBooleanToStr(tf)
if tf
    str = 'on';
else
    str = 'off';
end
end

function str = iDateTimeAsStringUsingDefaultLocalFormat(dt)
defaultFormat = datetime().Format;
dt.Format = defaultFormat;
str = char(dt);
end

function hSpacer = iAddFlexibleSpacer(parent)
hSpacer = uipanel(parent, 'BorderType', 'none');
hSpacer.HeightLimits = [0, Inf];
hSpacer.WidthLimits = [0, Inf];
end

function panel = iAddFixedHeightSpacer(parent, height)
panel = uipanel(parent, 'BorderType', 'none');
iSetPreferredHeight(panel, height);
iSetPreferredWidth(panel, 0);
end

function iAddFixedWidthSpacer(parent, width)
panel = uipanel(parent, 'BorderType', 'none');
iSetPreferredWidth(panel, width);
iSetPreferredHeight(panel, 0);
end

function iSetPreferredWidth(h, width)
h.WidthLimits = [width width];
end

function iSetPreferredHeight(h, height)
h.HeightLimits = [height height];
end

function iResizeAndCenterFigure(fig)
margin = 100;  
[screenWidth, screenHeight] = iGetScreenSize();
% subtract margin for task bar etc
fig.Position = [margin, margin, screenWidth-2*margin, screenHeight-2*margin];
end

function [width, height] = iGetScreenSize()
position = get(0, 'ScreenSize');
width = position(3);
height = position(4);
end

function pointsPerPixel = iGetPointsPerPixel()
pointsPerInch = 72;
pixelPerInch = get(groot, 'ScreenPixelsPerInch');
pointsPerPixel = pointsPerInch/pixelPerInch;
end

function margin = iTinyMargin()
margin = 0.0001;
end

function h = iFindFigure(h)
while not(isa(h, 'matlab.ui.Figure'))
    h = h.Parent;
end
end

function iChangeCursorOnMouseOverObject(object, figure, callbackdata)
if object == callbackdata.HitObject
    figure.Pointer = 'hand';
else
    figure.Pointer = 'arrow';
end
end

