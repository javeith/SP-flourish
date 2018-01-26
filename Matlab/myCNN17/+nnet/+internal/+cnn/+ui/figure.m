function f = figure(varargin)
% figure   Wrapper to customize the behavior of 'figure'

%   Copyright 2017 The MathWorks, Inc.

% We will respect the default user setting for the following figure
% properties:
%    'Color'
%    'Colormap'
%    'GraphicsSmoothing'
%    'InvertHardcopy'
%    'PaperOrientation'
%    'PaperPosition'
%    'PaperPositionMode'
%    'PaperSize'
%    'PaperType'
%    'PaperUnits'
%    'Renderer'
%    'RendererMode'
%
% The following figure properties are not applicable for the app:
%    'FileName'
%    'PointerShapeCData'
%    'PointerShapeHotSpot'
%    'UserData'
%
% The following properties we either customize or respect the user's
% settings:
%    'Position'
%
%
% ------ Setting the defaults for axes --------
%
% We will respect the default user setting for the following axes
% properties:
%    'ALim'
%    'ALimMode'
%    'AmbientLightColor'
%    'CLim'
%    'CLimMode'
%    'CameraPosition'
%    'CameraPositionMode'
%    'CameraTarget'
%    'CameraTargetMode'
%    'CameraUpVector'
%    'CameraUpVectorMode'
%    'CameraViewAngle'
%    'CameraViewAngleMode'
%    'Color'
%    'ColorOrder'
%    'ColorOrderIndex'
%    'DataAspectRatio'
%    'DataAspectRatioMode'
%    'GridAlpha'
%    'GridAlphaMode'
%    'GridColor'
%    'GridColorMode'
%    'GridLineStyle'
%    'LineStyleOrder'
%    'LineWidth'
%    'MinorGridAlpha'
%    'MinorGridAlphaMode'
%    'MinorGridColor'
%    'MinorGridColorMode'
%    'MinorGridLineStyle'
%    'Projection'
%    'XGrid'
%    'YGrid'
%    'ZGrid'
%
% The following axes properties are not applicable for the app:
%    'PlotBoxAspectRatio'
%    'TickDir'
%    'UserData'
%    'XColor'
%    'XLim'
%    'XTick'
%    'XTickLabel'
%    'YColor'
%    'YLim'
%    'YTick'
%    'YTickLabel'
%    'ZColor'
%    'ZLim'
%    'ZTick'
%    'ZTickLabel'
%
% The following properties we either customize or respect the user's
% settings:
%    'Position'
%
% 
% ------ Setting the defaults for text --------
%
% We will respect the default user setting for the following text
% properties:
%    'Color'
%
% The following text properties are not applicable for the app:
%    'UserData'

f = figure( ...
    ... The following should be overridden in the app:
    'Name', '', ...
    'Tag', '', ...
    ... For the following properties we pick sensible defaults, but they may be overriden in the app:
    'Alphamap', get(groot, 'factoryFigureAlphamap'), ...
    'BusyAction', 'queue', ...
    'Clipping', 'on', ...
    'HandleVisibility', 'callback', ...
    'IntegerHandle', 'off', ...
    'Interruptible', 'off', ...
    'NextPlot', 'add', ...
    'NumberTitle', 'off', ...
    'Pointer', 'arrow', ...
    'Resize', 'on', ...
    'UIContextMenu', [], ...
    'Units', 'pixels', ...
    'Visible', 'on', ...
    ... Figures docked to a toolgroup do not need the following settings, other figures may override in the app:
    'WindowStyle', 'normal', ...
    'DockControls', 'off', ...
    'MenuBar', 'none', ...
    'ToolBar', 'none', ...
    ... Set all callbacks to empty or factory:
    'ButtonDownFcn', {}, ...
    'CloseRequestFcn', get(groot, 'factoryFigureCloseRequestFcn'), ...
    'CreateFcn', {}, ...
    'DeleteFcn', get(groot, 'factoryFigureDeleteFcn'), ...
    'KeyPressFcn', {}, ...
    'KeyReleaseFcn', {}, ...
    'SizeChangedFcn', {}, ...
    'WindowButtonDownFcn', {}, ...
    'WindowButtonMotionFcn', {}, ...
    'WindowButtonUpFcn', {}, ...
    'WindowKeyPressFcn', {}, ...
    'WindowKeyReleaseFcn', {}, ...
    'WindowScrollWheelFcn', {}, ...
    ...
    ... Setting defaults for axes
    ...
    ... For the following properties we pick sensible defaults, but they may be overriden in the app:
    'defaultAxesActivePositionProperty', get(groot, 'factoryAxesActivePositionProperty'), ...
    'defaultAxesBox', get(groot, 'factoryAxesBox'), ...
    'defaultAxesBoxStyle', get(groot, 'factoryAxesBoxStyle'), ...
    'defaultAxesBusyAction', 'queue', ...
    'defaultAxesClipping', 'on', ...
    'defaultAxesClippingStyle', get(groot, 'factoryAxesClippingStyle'), ...
    'defaultAxesFontAngle', get(groot, 'factoryAxesFontAngle'), ...
    'defaultAxesFontName', get(groot, 'factoryAxesFontName'), ...
    'defaultAxesFontSize', get(groot, 'factoryAxesFontSize'), ...
    'defaultAxesFontSmoothing', get(groot, 'factoryAxesFontSmoothing'), ...
    'defaultAxesFontUnits', get(groot, 'factoryAxesFontUnits'), ...
    'defaultAxesFontWeight', get(groot, 'factoryAxesFontWeight'), ...
    'defaultAxesHandleVisibility', 'callback', ...
    'defaultAxesHitTest', 'on', ...
    'defaultAxesInterruptible', get(groot, 'factoryAxesInterruptible'), ...
    'defaultAxesLabelFontSizeMultiplier', get(groot, 'factoryAxesLabelFontSizeMultiplier'), ...
    'defaultAxesLayer', get(groot, 'factoryAxesLayer'), ...
    'defaultAxesNextPlot', get(groot, 'factoryAxesNextPlot'), ...
    'defaultAxesPickableParts', get(groot, 'factoryAxesPickableParts'), ...
    'defaultAxesPlotBoxAspectRatioMode', 'auto', ...
    'defaultAxesSelectionHighlight', 'on', ...
    'defaultAxesSortMethod', 'depth', ...
    'defaultAxesTickDirMode', 'auto', ...
    'defaultAxesTickLabelInterpreter', 'none', ...
    'defaultAxesTickLength', get(groot, 'factoryAxesTickLength'), ...
    'defaultAxesTitleFontSizeMultiplier', get(groot, 'factoryAxesTitleFontSizeMultiplier'), ...
    'defaultAxesTitleFontWeight', get(groot, 'factoryAxesTitleFontWeight'), ...
...    'defaultAxesUIContextMenu', [], ... % Cannot set default UI context menu. But it is empty anyway
    'defaultAxesUnits', get(groot, 'factoryAxesUnits'), ...
    'defaultAxesView', get(groot, 'factoryAxesView'), ...
    'defaultAxesVisible', 'on', ...
    'defaultAxesXAxisLocation', get(groot, 'factoryAxesXAxisLocation'), ...
    'defaultAxesXColorMode', 'auto', ...
    'defaultAxesXDir', 'normal', ...
    'defaultAxesXLimMode', 'auto', ...
    'defaultAxesXMinorGrid', get(groot, 'factoryAxesXMinorGrid'), ...
    'defaultAxesXMinorTick', get(groot, 'factoryAxesXMinorTick'), ...
    'defaultAxesXScale', get(groot, 'factoryAxesXScale'), ...
    'defaultAxesXTickLabelMode', 'auto', ...
    'defaultAxesXTickLabelRotation', get(groot, 'factoryAxesXTickLabelRotation'), ...
    'defaultAxesXTickMode', 'auto', ...
    'defaultAxesYAxisLocation', get(groot, 'factoryAxesYAxisLocation'), ...
    'defaultAxesYColorMode', 'auto', ...
    'defaultAxesYDir', 'normal', ...
    'defaultAxesYLimMode', 'auto', ...
    'defaultAxesYMinorGrid', get(groot, 'factoryAxesYMinorGrid'), ...
    'defaultAxesYMinorTick', get(groot, 'factoryAxesYMinorTick'), ...
    'defaultAxesYScale', get(groot, 'factoryAxesYScale'), ...
    'defaultAxesYTickLabelMode', 'auto', ...
    'defaultAxesYTickLabelRotation', get(groot, 'factoryAxesYTickLabelRotation'), ...
    'defaultAxesYTickMode', 'auto', ...
    'defaultAxesZColorMode', 'auto', ...
    'defaultAxesZDir', 'normal', ...
    'defaultAxesZLimMode', 'auto', ...
    'defaultAxesZMinorGrid', get(groot, 'factoryAxesZMinorGrid'), ...
    'defaultAxesZMinorTick', get(groot, 'factoryAxesZMinorTick'), ...
    'defaultAxesZScale', get(groot, 'factoryAxesZScale'), ...
    'defaultAxesZTickLabelMode', 'auto', ...
    'defaultAxesZTickLabelRotation', get(groot, 'factoryAxesZTickLabelRotation'), ...
    'defaultAxesZTickMode', 'auto', ...
    ... Set all callbacks to empty:
    'defaultAxesButtonDownFcn', {}, ...
    'defaultAxesCreateFcn', {}, ...
    'defaultAxesDeleteFcn', {}, ...
    ...
    ... Setting defaults for text
    ...
    ... For the following properties we pick sensible defaults, but they may be overriden in the app:
    'defaultTextBackgroundColor', 'none', ...
    'defaultTextBusyAction', 'queue', ...
    'defaultTextClipping', 'off', ...
    'defaultTextEdgeColor', 'none', ...
    'defaultTextEditing', 'off', ...
    'defaultTextFontAngle', get(groot, 'factoryTextFontAngle'), ...
    'defaultTextFontName', get(groot, 'factoryTextFontName'), ...
    'defaultTextFontSize', get(groot, 'factoryTextFontSize'), ...
    'defaultTextFontSmoothing', get(groot, 'factoryTextFontSmoothing'), ...
    'defaultTextFontUnits', get(groot, 'factoryTextFontUnits'), ...
    'defaultTextFontWeight', get(groot, 'factoryTextFontWeight'), ...
    'defaultTextHandleVisibility', get(groot, 'factoryTextHandleVisibility'), ...
    'defaultTextHitTest', 'on', ...
    'defaultTextHorizontalAlignment', get(groot, 'factoryTextHorizontalAlignment'), ...
    'defaultTextInterpreter', 'none', ...
    'defaultTextInterruptible', 'off', ...
    'defaultTextLineStyle', get(groot, 'factoryTextLineStyle'), ...
    'defaultTextLineWidth', get(groot, 'factoryTextLineWidth'), ...
    'defaultTextMargin', get(groot, 'factoryTextMargin'), ...
    'defaultTextPickableParts', get(groot, 'factoryTextPickableParts'), ...
    'defaultTextRotation', 0, ...
    'defaultTextSelectionHighlight', get(groot, 'factoryTextSelectionHighlight'), ...
...    'defaultTextUIContextMenu', [], ... % Cannot set default UI context menu. But it is empty anyway
    'defaultTextUnits', get(groot, 'factoryTextUnits'), ...
    'defaultTextVerticalAlignment', get(groot, 'factoryTextVerticalAlignment'), ...
    'defaultTextVisible', 'on', ...
    ... Set all callbacks to empty:
    'defaultTextButtonDownFcn', {}, ...
    'defaultTextCreateFcn', {}, ...
    'defaultTextDeleteFcn', {}, ...
    varargin{:});
end
