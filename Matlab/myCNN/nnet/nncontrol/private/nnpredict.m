function nnpredict(cmd,arg1,arg2,arg3)
%NNPREDICT Neural Network Predictive Controller GUI for Neural Network Controller Toolbox.
%
%  Synopsis
%
%    nnpredict(cmd,arg1,arg2,arg3)
%
%  Warning!!
%
%    This function may be altered or removed in future
%    releases of Neural Network Toolbox. We recommend
%    you do not write code which calls this function.
%    This function is generally being called from a Simulink block.

% Orlando De Jesus, Martin Hagan, 1-25-00
% Copyright 1992-2011 The MathWorks, Inc.


% CONSTANTS
me = 'Neural Network Predictive Control';

% DEFAULTS
if nargin == 0, cmd = ''; else cmd = lower(cmd); end

% FIND WINDOW IF IT EXISTS
% 9/3/99 We alow the program to see hidden handles
fig = findall(0,'type','figure','name',me);
if ~isempty(fig) && isempty(get(fig,'children')), fig = []; end

% GET WINDOW DATA IF IT EXISTS
if ~isempty(fig)
  H = get(fig,'userdata');
  
  if strcmp(cmd,'')
    if get(H.gcbh_ptr,'userdata')~=arg1
      delete(fig);
      fig= [];
    end
  else
     % ODJ 1-13-00 We check if the field SimulationStatus exist before reading that field
    if isfield(get(H.gcbh_ptr),'UserData')
       if isfield(get_param(get_param(get(H.gcbh_ptr,'userdata'),'parent'),'objectparameters'),'SimulationStatus')
          SimulationStatus=get_param(get_param(get(H.gcbh_ptr,'userdata'),'parent'),'SimulationStatus');
       else
          SimulationStatus='none';
       end
    else
       SimulationStatus='none';
    end
    if (strcmp(SimulationStatus,'running') | strcmp(SimulationStatus,'paused')) & ~strcmp(cmd,'close')
      set(H.error_messages(1,1),'string','You must stop the simulation to change NN configuration parameters.');
      return;
    end
  end
end

%==================================================================
% Activate the window.
%
% ME() or ME('')
%==================================================================

if strcmp(cmd,'')
  if ~isempty(fig)
    figure(fig)
    set(fig,'visible','on')
  else
    nncontrolutil('nnpredict','init',arg1,arg2);
  end

%==================================================================
% Close the window.
%
% ME() or ME('')
%==================================================================

elseif strcmp(cmd,'close') && ~isempty(fig)
   delete(fig)
   return;

elseif (strcmp(cmd,'apply') | strcmp(cmd,'ok')) && ~isempty(fig)
  arg1=get(H.gcbh_ptr,'userdata');
  
  a1 = str2num(get(H.N2_edit,'string'));
  if length(a1) == 0, a1=0; end
  if ~sanitycheckparam(a1) | a1<2,
     N2=get_param(arg1,'N2'); 
     present_error(H,H.N2_edit,N2,1, ...
         'Please, correct the cost horizon value (N2)'); 
     return
  else set_param(arg1,'N2',num2str(a1)); end
  
  a1 = str2num(get(H.Nu_edit,'string'));
  if length(a1) == 0, a1=0; end
  if ~sanitycheckparam(a1) | a1<2,
     Nu=get_param(arg1,'Nu'); 
     present_error(H,H.Nu_edit,Nu,1, ...
         'Please, correct the control horizon value (Nu)'); 
     return
  else set_param(arg1,'Nu',num2str(a1)); end
    
  a1 = str2num(get(H.rho_edit,'string'));
  if ~sanitycheckparam(a1),
     rho=get_param(arg1,'rho'); 
     present_error(H,H.rho_edit,rho,1, ...
         'Please, correct the control weighting factor'); 
     return
  else set_param(arg1,'rho',num2str(a1)); end
    
  a1 = str2num(get(H.alpha_edit,'string'));
  if ~sanitycheckparam(a1),
     alpha=get_param(arg1,'alpha');
     present_error(H,H.alpha_edit,alpha,1, ...
         'Please, correct the search parameter'); 
     return
  else set_param(arg1,'alpha',num2str(a1)); end
    
  func_index=['csrchgol';'csrchbac';'csrchhyb';'csrchbre';'csrchcha'];
  a1 = get(H.csrchfun_edit,'value');
  if (a1 < 1) | (a1 > 5), 
     csrchfun=get_param(arg1,'csrchfun'); 
     a1=nnstring.first_match(csrchfun,func_index);
     set(H.csrchfun_edit,'value',a1);
     present_error(H,H.csrchfun_edit,a1,0, ...
        'Please, correct the minimization function'); 
     return
  else set_param(arg1,'csrchfun',func_index(a1,:)); end
    
  a1 = str2num(get(H.maxiter_edit,'string'));
  if length(a1) == 0, a1=0; end
  if ~sanitycheckparam(a1) | a1<1,
     maxiter=get_param(arg1,'maxiter');
     present_error(H,H.maxiter_edit,maxiter,1, ...
         'Please, correct the number of iterations per sample time'); 
     return
  else set_param(arg1,'maxiter',num2str(a1)); end
    
  if strcmp(cmd,'ok')
     delete(fig)
  else
    set(H.error_messages(1,1),'string',sprintf(' '));
  end
  
%==================================================================
% Execute Training.
%
% ME('training')
%==================================================================

elseif strcmp(cmd,'training') && ~isempty(fig)
  arg1=get(H.gcbh_ptr,'userdata');
  arg2=get(H.gcb_ptr,'userdata');
  nnident('',arg1,arg2,'nnpredict');

%==================================================================
% Initialize the window.
%
% ME('init')
%==================================================================

elseif strcmp(cmd,'init') && isempty(fig)

  % 1-13-00 ODJ We check if the system is locked.
  sys_par=arg2;
  sys_par2=arg2;
  while ~isempty(sys_par2)
      sys_par=sys_par2;
      sys_par2=get_param(sys_par,'parent');
  end
  if strcmp('on',get_param(sys_par,'lock'))
      window_en='off';
  else
      window_en='on';
  end
  
  stdunits = 'character';
  uipos = getuipositions;

  fig = figure('CloseRequestFcn','nncontrolutil(''nnpredict'',''close'')', ...
   'Interruptible','off', ...
   'BusyAction','cancel', ...
   'HandleVis','Callback', ...
   'Color',[0.8 0.8 0.8], ...
  'MenuBar','none', ...
   'Name',me, ...
   'numbertitle','off', ...
   'IntegerHandle',  'off',...
   'Units', 'character',...
  'PaperUnits','points', ...
  'Position',uipos.fig, ...
  'Tag','Fig4', ...
  'Resize','off', ... 
  'ToolBar','none');
  frame1 = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
  'ListboxTop',0, ...
  'Position',uipos.frame1, ...
  'Style','frame', ...
  'Tag','Frame1');
  frame2 = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
  'ListboxTop',0, ...
  'Position',uipos.frame2, ...
  'Style','frame', ...
  'Tag','Frame2');
  frame3 = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
  'ListboxTop',0, ...
  'Position',uipos.frame3, ...
  'Style','frame', ...
  'Tag','Frame3');
  H.Title_nnpredict = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
  'FontSize',14, ...
  'ListboxTop',0, ...
  'Position',uipos.Title_nnpredict, ...
  'String','Neural Network Predictive Control', ...
  'Style','text', ...
  'Tag','Title_nnpredict');
  H.Train_NN = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
  'Callback','nncontrolutil(''nnpredict'',''training'');', ...
    'Enable',window_en, ...
  'ListboxTop',0, ...
  'Position',uipos.Train_NN, ...
  'String','Plant Identification', ...
  'ToolTipStr','Opens a window where you can develop the neural network plant model.',...
  'Tag','Pushbutton1');
  H.OK_but = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
  'Callback','nncontrolutil(''nnpredict'',''ok'')', ...
    'Enable',window_en, ...
  'ListboxTop',0, ...
  'Position',uipos.OK_but, ...
  'String','OK', ...
   'ToolTipStr','Save the parameters into the neural network controller block and close this window.',...
  'Tag','OK_but');
  H.Cancel_but = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
  'Callback','nncontrolutil(''nnpredict'',''close'')', ...
  'ListboxTop',0, ...
  'Position',uipos.Cancel_but, ...
  'String','Cancel', ...
   'ToolTipStr','Discard the neural network controller parameters.',...
  'Tag','Pushbutton1');
  H.Apply_but = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
  'Callback','nncontrolutil(''nnpredict'',''apply'')', ...
    'Enable',window_en, ...
  'ListboxTop',0, ...
  'Position',uipos.Apply_but, ...
  'String','Apply', ...
   'ToolTipStr','Save the parameters into the neural network controller block.',...
  'Tag','Apply_but');
  H.N2_text = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
    'Enable',window_en, ...
   'HorizontalAlignment','right', ...
  'ListboxTop',0, ...
  'Position',uipos.N2_text, ...
  'String','Cost Horizon (N2) ', ...
  'Style','text', ...
   'ToolTipStr','Horizon over which the set-point error is minimized.',...
  'Tag','StaticText2');  
  H.N2_edit = uicontrol('Parent',fig, ...
    'Units','character', ...
    'BackgroundColor',[1 1 1], ...
    'Enable',window_en, ...
    'ListboxTop',0, ...
    'Position',uipos.N2_edit, ...
    'String','', ...
    'Style','edit', ...
  'Callback',['nncontrolutil(''nnpredict'',''check_params'',''N2_edit'', ''', get(H.N2_text, 'String'),''');'], ...
    'ToolTipStr','Cost horizon value for the set-point error.',...
    'Tag','N2_edit');
  H.Nu_text = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
    'Enable',window_en, ...
   'HorizontalAlignment','right', ...
  'ListboxTop',0, ...
  'Position',uipos.Nu_text, ...
  'String','Control Horizon (Nu) ', ...
  'Style','text', ...
   'ToolTipStr','Horizon over which the deviation in control action is minimized.',...
  'Tag','StaticText2');
  H.Nu_edit = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[1 1 1], ...
    'Enable',window_en, ...
  'ListboxTop',0, ...
  'Position',uipos.Nu_edit, ...
  'String','2', ...
  'Style','edit', ...
  'Callback',['nncontrolutil(''nnpredict'',''check_params'',''Nu_edit'', ''', get(H.Nu_text, 'String'),''');'], ...
   'ToolTipStr','Horizon over which the deviation in control action is minimized.',...
  'Tag','Nu_edit');
  H.rho_text2 = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
    'Enable',window_en, ...
   'HorizontalAlignment','right', ...
  'ListboxTop',0, ...
  'Position',uipos.rho_text2, ...
  'String','Control Weighting Factor (', ...
  'Style','text', ...
   'ToolTipStr','Control weighting factor that multiplies the deviation in control action.',...
  'Tag','StaticText2');
  H.rho_text = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
    'Enable',window_en, ...
  'FontSize',10, ...
  'ListboxTop',0, ...
  'Position',uipos.rho_text, ...
  'String','\rho', ...
  'Style','text', ...
   'ToolTipStr','Control weighting factor that multiplies the deviation in control action.',...
  'Tag','StaticText2');
  H.rho_text3 = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
    'Enable',window_en, ...
   'HorizontalAlignment','left', ...
  'ListboxTop',0, ...
  'Position',uipos.rho_text3, ...
  'String',')', ...
  'Style','text', ...
   'ToolTipStr','Control weighting factor that multiplies the deviation in control action.',...
  'Tag','StaticText2');
  H.rho_edit = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[1 1 1], ...
    'Enable',window_en, ...
  'ListboxTop',0, ...
  'Position',uipos.rho_edit, ...
  'String','0.03', ...
  'Style','edit', ...
  'Callback','nncontrolutil(''nnpredict'',''check_params'',''rho_edit'', ''Control Weighting Factor'');', ...
   'ToolTipStr','Control weighting factor that multiplies the deviation in control action.',...
  'Tag','rho_edit');
  H.alpha_text2 = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
    'Enable',window_en, ...
   'HorizontalAlignment','right', ...
  'ListboxTop',0, ...
  'Position',uipos.alpha_text2, ...
  'String','Search Parameter (', ...
  'Style','text', ...
   'ToolTipStr','Search parameter to be used in the minimization routine.',...
  'Tag','StaticText2');
  H.alpha_text = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
    'Enable',window_en, ...
  'FontSize',10, ...
  'ListboxTop',0, ...
  'Position',uipos.alpha_text, ...
  'String','\alpha', ...
  'Style','text', ...
   'ToolTipStr','Search parameter to be used in the minimization routine.',...
  'Tag','StaticText2');
  H.alpha_text3 = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
    'Enable',window_en, ...
   'HorizontalAlignment','left', ...
  'ListboxTop',0, ...
  'Position',uipos.alpha_text3, ...
  'String',')', ...
  'Style','text', ...
   'ToolTipStr','Search parameter to be used in the minimization routine.',...
  'Tag','StaticText2');
  H.alpha_edit = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[1 1 1], ...
    'Enable',window_en, ...
  'ListboxTop',0, ...
  'Position',uipos.alpha_edit, ...
  'String','0.001', ...
  'Style','edit', ...
  'Callback',['nncontrolutil(''nnpredict'',''check_params'',''alpha_edit'', ''Search Parameter'');'], ...
   'ToolTipStr','Search parameter to be used in the minimization routine.',...
  'Tag','EditText1');
  H.csrchfun_edit = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[1 1 1], ...
    'Enable',window_en, ...
  'ListboxTop',0, ...
  'Max',5, ...
  'Position',uipos.csrchfun_edit, ...
  'String',['csrchgol';'csrchbac';'csrchhyb';'csrchbre';'csrchcha'], ...
  'Style','popupmenu', ...
  'Tag','PopupMenu1', ...
  'Value',1);
  H.csrchfun_text = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
    'Enable',window_en, ...
   'HorizontalAlignment','right', ...
  'ListboxTop',0, ...
  'Position',uipos.csrchfun_text, ...
  'String','Minimization Routine', ...
  'Style','text', ...
   'ToolTipStr','Line search routine to be used in the optimization algorithm.',...
  'Tag','StaticText3');
  H.maxiter_text = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
    'Enable',window_en, ...
   'HorizontalAlignment','right', ...
  'ListboxTop',0, ...
  'Position',uipos.maxiter_text, ...
  'String','Iterations Per Sample Time', ...
  'Style','text', ...
   'ToolTipStr','Maximum number of iterations of the optimization algorithm per sample time.',...
   'Tag','StaticText2');  
H.maxiter_edit = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[1 1 1], ...
    'Enable',window_en, ...
  'ListboxTop',0, ...
  'Position',uipos.maxiter_edit, ...
  'String','2', ...
  'Style','edit', ...
  'Callback',['nncontrolutil(''nnpredict'',''check_params'',''maxiter_edit'', ''', get(H.maxiter_text, 'String'),''');'], ...
  'ToolTipStr','Maximum number of iterations of the optimization algorithm per sample time.',...
  'Tag','maxiter_edit');
  frame4 = uicontrol('Parent',fig, ...
  'Units','character', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
  'ListboxTop',0, ...
  'Position',uipos.frame4, ...
  'Style','frame', ...
  'Tag','Frame4');
  H.error_messages(1,1)= uicontrol('Parent',fig, ...
  'Units','character', ...
   'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
  'FontWeight','bold', ...
  'ForegroundColor',[0 0 1], ...
  'ListboxTop',0, ...
  'Position',uipos.error_messages, ...
  'Style','text', ...
  'ToolTipStr','Feedback line with important messages for the user.',...
  'Tag','StaticText1');

% We create the menus for the block.
  H.Handles.Menus.File.Top= uimenu('Parent',fig, ...
  'Label','File');
  H.Handles.Menus.File.ImportModel = uimenu('Parent',...
   H.Handles.Menus.File.Top,...
  'Label','Import Network...',...
  'Accelerator','I',...
  'Callback','nncontrolutil(''nnimport'',''init'',gcbf,''nnpredict'',''nnpredict'');',...
    'Enable',window_en, ...
   'Tag','ImportModel');
  H.Handles.Menus.File.Export = uimenu('Parent',H.Handles.Menus.File.Top, ...
   'Label','Export Network...', ...
   'Accelerator','E', ...
   'Callback','nncontrolutil(''nnexport'',''init'',gcbf,''nnpredict'',''nnpredict'')', ...
    'Enable',window_en, ...
   'Tag','ExportMenu');
  H.Handles.Menus.File.Save_NN = uimenu('Parent',...
   H.Handles.Menus.File.Top,...
   'Label','Save',...
   'Separator','on', ...
   'Accelerator','S',...
   'Callback','nncontrolutil(''nnpredict'',''apply'');',...
    'Enable',window_en, ...
   'Tag','ImportModel');
  H.Handles.Menus.File.Save_Exit_NN = uimenu('Parent',...
   H.Handles.Menus.File.Top,...
   'Label','Save and Exit',...
   'Accelerator','A',...
   'Callback','nncontrolutil(''nnpredict'',''ok'');',...
    'Enable',window_en, ...
   'Tag','ImportModel');
  H.Handles.Menus.File.Close = uimenu('Parent',H.Handles.Menus.File.Top, ...
   'Callback','nncontrolutil(''nnpredict'',''close'',gcbf);', ...
   'Separator','on', ...
   'Label','Exit', ...
   'Accelerator','X', ...
   'Tag','CloseMenu');

  H.Handles.Menus.Window.Top = matlab.ui.internal.createWinMenu(fig);

  H.Handles.Menus.Help.Top = uimenu('Parent',fig, ...
   'Label','Help');
  H.Handles.Menus.Help.Main = uimenu('Parent',H.Handles.Menus.Help.Top, ...
   'Label','Main Help', ...
   'Callback','nncontrolutil(''nnpredicthelp'',''main'');',...
   'Accelerator','H');
  H.Handles.Menus.Help.PlantIdent = uimenu('Parent',H.Handles.Menus.Help.Top, ...
   'Label','Plant Identification...', ...
   'CallBack','nncontrolutil(''nnpredicthelp'',''plant_ident'');');
  H.Handles.Menus.Help.Simulation = uimenu('Parent',H.Handles.Menus.Help.Top, ...
   'Label','Simulation...', ...
   'Separator','on',...
   'CallBack','nncontrolutil(''nnpredicthelp'',''simulation'');');

  H.gcbh_ptr = uicontrol('Parent',fig,'visible','off');
  set(H.gcbh_ptr,'userdata',arg1);
  H.gcb_ptr = uicontrol('Parent',fig,'visible','off');
  set(H.gcb_ptr,'userdata',arg2);
  
  N2=get_param(arg1,'N2'); 
  set(H.N2_edit,'string',num2str(N2));
    
  Nu=get_param(arg1,'Nu'); 
  set(H.Nu_edit,'string',num2str(Nu));
       
  rho=get_param(arg1,'rho'); 
  set(H.rho_edit,'string',num2str(rho));
    
  alpha=get_param(arg1,'alpha'); 
  set(H.alpha_edit,'string',num2str(alpha));
    
  func_index=['csrchgol';'csrchbac';'csrchhyb';'csrchbre';'csrchcha'];
  csrchfun=get_param(arg1,'csrchfun'); 
  vv=nnstring.first_match(csrchfun,func_index);
  set(H.csrchfun_edit,'value',vv);
    
  maxiter=get_param(arg1,'maxiter'); 
  set(H.maxiter_edit,'string',maxiter);
  set(fig,'userdata',H)
  
  set(H.error_messages(1,1),'string',sprintf('Perform plant identification before controller configuration.'));
  
elseif strcmp(cmd,'check_params')
    
    checkparam(arg1, H, arg2);    
  
end

function present_error(H,text_field,field_value,field_type,message1)

if text_field~=0
   if field_type      % Number
      set(text_field,'string',num2str(field_value));
   else               % ASCII or No change.
      set(text_field,'string',field_value);
   end
else
   text_field=0;
end   
set(H.error_messages,'string',message1);
errordlg(message1,'Plant Identification Warning','modal');




function paramok = checkparam(param2check, handles, varargin)

paramok = true; %set to true initially
throwerrdlg = true; % throw error dialog

if nargin > 2
    paramlabel = varargin{1};
else
    paramlabel = '';
    throwerrdlg = false;
end

paramH = getfield(handles, param2check);
paramval = str2num(get(paramH, 'String'));

try
    % Common Checks for all params
    message1 = sprintf('Illegal value assigned to parameter');

    if ~sanitycheckparam(paramval)
        nnerr.throw('Parameters',message1);
    end

    % Param specific checks
    switch lower(param2check)         
        case {'N2_edit', 'Nu_edit', 'rho_edit', 'alpha_edit', 'maxiter_edit'}
            % no specific checks for these parameters.
            % no-op case incorporated for these to make it easy for
            % extension later
    end
    
catch
    if throwerrdlg
        message1 = sprintf('Illegal value assigned to ''%s'' parameter', paramlabel);
        errordlg(message1,'Plant Identification Warning','modal');
    end
    paramok = false;
end





function paramok = sanitycheckparam(param)

if isempty(param) || iscell(param) ...
    || ~isscalar(param) || ~isnumeric(param) ...
        || ~isfinite(param) || ~isreal(param)        
    paramok = false;
    return;
end

paramok = true;





function uipos = getuipositions


sunits = get(0, 'Units');
set (0, 'Units', 'character');
ssinchar = get(0, 'ScreenSize');
set (0, 'Units', sunits);


editw = 12;
labelw = 33;
border = 1.3333;
framew = editw + (3*border) + labelw;
edith = 1.53846;
butwbig = 30;
butwsmall = (2*framew)-(border+butwbig);

if butwsmall > 12
    butwsmall = 12;
end

buth = 1.65;


figw = (2*framew) + (3*border);
figh = 17.8462;
figl = (ssinchar(3) - figw) / 2;
figb = (ssinchar(4) - figh) / 2;

uipos.fig = [figl,figb,figw,figh];


uipos.frame2 = [border,9.12821,framew,5.15385];
uipos.frame3 = [(border*2)+framew,9.12821,framew,5.15385];
uipos.frame1 = [border,6.5641,(2*framew + border),2.35897];
uipos.frame4 = [border,1.23077,(2*framew + border),2.25641];

uipos.N2_text = [2*border,12.1846,labelw,edith];
uipos.Nu_text = [2*border,9.9538,labelw,edith];


uipos.N2_edit = [uipos.N2_text(1)+labelw+border,12.3846,editw,edith];
uipos.Nu_edit = [uipos.N2_edit(1),10.1538,editw,edith];


uipos.maxiter_text  = [uipos.N2_edit(1)+editw+(2*border),6.8,labelw,edith];
uipos.rho_edit = [uipos.maxiter_text(1)+labelw+border,12.3846,editw,edith];
uipos.alpha_edit = [uipos.rho_edit(1),10.1538,editw,edith];
uipos.maxiter_edit = [uipos.rho_edit(1),7,editw,edith];

uipos.rho_text3 = [uipos.maxiter_edit(1)-(border+1.06667),12.1846,1.5,edith];
uipos.rho_text = [uipos.rho_text3(1)-1.86667,12.3846,1.86667,edith];
labelsw = labelw - (uipos.rho_text3(3)+uipos.rho_text(3));
uipos.rho_text2 = [uipos.rho_text(1)-labelsw,12.1846,labelsw,edith];

uipos.alpha_text3 = [uipos.rho_text3(1),9.9538,1.5,edith];
uipos.alpha_text = [uipos.alpha_text3(1)-1.86667,10.1538,1.86667,edith];
uipos.alpha_text2 = [uipos.alpha_text(1)-labelsw,9.9538,labelsw,edith];

uipos.csrchfun_edit = [uipos.maxiter_text(1)-((2*border)+editw+4),7,editw+4,edith];
uipos.csrchfun_text = [uipos.csrchfun_edit(1)-(border+labelw-4),6.8,labelw-4,edith];

uipos.Train_NN = [border*2,4.20513,butwbig,buth];

uipos.Apply_but = [figw-((2*border)+butwsmall),4.20513,butwsmall,buth];
uipos.Cancel_but = [uipos.Apply_but(1)-(border+butwsmall),4.20513,butwsmall,buth];
uipos.OK_but = [uipos.Cancel_but(1)-(border+butwsmall),4.20513,butwsmall,buth];

uipos.error_messages = [border+(0.3*border),border,(2*framew)+border-(0.6*border),2.05128];
tlw = (2*framew) + border;
uipos.Title_nnpredict = [border,14.6154,tlw,2.23077];









