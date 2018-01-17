function nnd12sd1(cmd,arg1)
%NND12SD1 Steepest descent backpropagation demonstration #1.
%
%  Select two network weight/bias parameters to train and their
%  initial values, and watch as the network is trained with steepest
%  descent backpropagation to find a minimum on the error surface. 
%
%  This demonstration requires Neural Network Toolbox.

% Copyright 1994-2011 Martin T. Hagan
% First Version, 8-31-95.

%==================================================================

% CONSTANTS
me = 'nnd12sd1';
max_t = 0.5;
w_max = 10;
p_max = 2;
circle_size = 6;

% FLAGS
change_func = 0;

% DEFAULTS
if nargin == 0, cmd = ''; else cmd = lower(cmd); end

% FIND WINDOW IF IT EXISTS
fig = nndfgflg(me);
if ~isempty(fig) && isempty(get(fig,'children')), fig = []; end
  
% GET WINDOW DATA IF IT EXISTS
if ~isempty(fig)
  H = get(fig,'userdata');
  fig_axis = H(1);            % window axis
  desc_text = H(2);           % handle to first line of text sequence
  surf_axis = H(3);           % error surface axis
  cont_axis = H(4);           % error contour axis
  surf_ptr = H(5);            % pointer to error surface
  cont_ptr = H(6);            % pointer to error contour handles
  variables = H(7:10);        % variable name texts
  radios = H(11:13);          % radio buttons
  option_ptr = H(14);         % index of active radio
  path_ptr = H(15);           % pointer to training path handles
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
    feval(me,'init')
  end

%==================================================================
% Close the window.
%
% ME() or ME('')
%==================================================================

elseif strcmp(cmd,'close') & ~isempty(fig)
  delete(fig)

%==================================================================
% Initialize the window.
%
% ME('init')
%==================================================================

elseif strcmp(cmd,'init') & isempty(fig)

  % CHECK FOR NNT
  if ~nntexist(me), return, end

  % CONSTANTS
  W1 = [10; 10];
  b1 = [-5;5];
  W2 = [1 1];
  b2 = [-1];
  P = -2:0.1:2;
  [R,Q] = size(P);
  A1 = logsig(W1*P+b1*ones(1,Q));
  T = logsig(W2*A1+b2*ones(1,Q));

  % NEW DEMO FIGURE
  fig = nndemof2(me,'DESIGN','Steepest Descent Backprop #1','','Chapter 12');
  set(fig, ...
    'windowbuttondownfcn',nncallbk(me,'down'), ...
    ... % 'Backing_Store','off',...
    'nextplot','add');
  H = get(fig,'userdata');
  fig_axis = H(1);
  desc_text = H(2);

  % ICON
  nndicon(12,458,363,'shadow')

  % NETWORK POSITIONS
  x1 = 40;     % input
  x2 = x1+90;  % 1st layer sum
  x3 = x2+40;  % 1st layer transfer function
  x4 = x3+100; % 2nd layer sum
  x5 = x4+40;  % 2nd layer transfer function
  x6 = x5+50;  % output
  y1 = 330;    % top neuron
  y2 = y1-35;  % input & output neuron
  y3 = y1-70;  % bottom neuron
  sz = 15;     % size of icons
  wx = 55;     % weight vertical offset (from 1st layer)
  wy = 40;     % weight horizontal offset (from middle)

  % NETWORK INPUT
  nndtext(x1-10,y2,'p');
  plot([x2-sz x1 x2-sz],[y1 y2 y3],'linewidth',2,'color',nnred);

  % TOP NEURON
  plot([x2 x2 x3],[y1-sz*2 y1 y1],'linewidth',2,'color',nnred);
  nndsicon('sum',x2,y1,sz)
  nndsicon('logsig',x3,y1,sz)
  var1 = nndtext(x2-wx,y2+wy,'W1(1,1)');
  set(var1,'color',[1 1 1])
  nndtext(x2+2,y1-sz*2-13,'1');
  var2 = nndtext(x2+10,y1-sz*2,'b1(1)','left');

  % BOTTOM NEURON
  plot([x2 x2 x3],[y3-sz*2 y3 y3],'linewidth',2,'color',nnred);
  nndsicon('sum',x2,y3,sz)
  nndsicon('logsig',x3,y3,sz)
  nndtext(x2-wx,y2-wy,'W1(2,1)');
  nndtext(x2+2,y3-sz*2-13,'1');
  var3 = nndtext(x2+10,y3-sz*2,'b1(2)','left');

  % OUTPUT NEURON
  plot([x3+sz x4-10 x3+sz],[y1 y2 y3],'linewidth',2,'color',nnred);
  plot([x4 x4 x6],[y2-sz*2 y2 y2],'linewidth',2,'color',nnred);
  plot([x6-10 x6 x6-10],[y2-7 y2 y2+7],'linewidth',2,'color',nnred);
  nndsicon('sum',x4,y2,sz)
  nndsicon('logsig',x5,y2,sz);
  var4 = nndtext(x3+wx,y2+wy,'W2(1,1)');
  set(var4,'color',[1 1 1])
  nndtext(x3+wx,y2-wy,'W2(1,2)');
  nndtext(x4+2,y2-sz*2-13,'1');
  nndtext(x4+10,y2-sz*2,'b2','left');
  nndtext(x5+sz+5,y2+8,'a2','left');

  % RADIO BUTTONS
  option = 1;
  radio1 = uicontrol(...
    'units','points',...
    'position',[20 180 130 20],...
    'style','radio',...
    'string','W1(1,1), W2(1,1)',...
    'callback',[me '(''radio'',1)'],...
    'back',nnltgray,...
    'value',1);
  radio2 = uicontrol(...
    'units','points',...
    'position',[155 180 115 20],...
    'style','radio',...
    'string','W1(1,1), b1(1)',...
    'back',nnltgray,...
    'callback',[me '(''radio'',2)']);
  radio3 = uicontrol(...
    'units','points',...
    'position',[270 180 105 20],...
    'style','radio',...
    'string','b1(1), b1(2)',...
    'back',nnltgray,...
    'callback',[me '(''radio'',3)']);

  % ERROR SURFACE
  load nndbp1

  cont_axis = nnsfo('a2','',v1,v2,'');
  set(cont_axis, ...
    'units','points',...
    'position',[230 40 130 130],...
    'color',nnltyell,...
    'xlim',range1,...
    'ylim',range2,...
    'colororder',[0 0 0])
  [dummy,cont_h] = contour(x2,y2,E2,levels);
  set(cont_h,'CreateFcn','');
  plot3(range1([1 2 2 1 1]),range2([1 1 2 2 1]),1000*ones(1,5),...
    'color',nndkblue);
  cont_h2 = plot(optx,opty,'+','color',nnred);
  cont_h = [cont_h; cont_h2];
  view(2)

  surf_axis = nnsfo('a2','',v1,v2,'Sum Sq. Error');
  set(surf_axis, ...
    'units','points',...
    'position',[50 40 130 130],...
    'color',nnltyell,...
    'xlim',range1,...
    'ylim',range2)
  view(vw)
  set(gcf,'renderer','p');
  surf_h = surf(x1,y1,E1);
  set(surf_h,'CreateFcn','','facecolor','interp')
  
  colormap(bone)
  brighten(0.6)

  % BUTTONS
  drawnow % Let everything else appear before buttons 
  uicontrol(...
    'units','points',...
    'position',[400 110 60 20],...
    'string','Contents',...
    'callback','nndtoc')
  uicontrol(...
    'units','points',...
    'position',[400 75 60 20],...
    'string','Close',...
    'callback',[me '(''close'')'])

  % DATA POINTERS
  surf_ptr = uicontrol('visible','off','userdata',surf_h);
  cont_ptr = uicontrol('visible','off','userdata',cont_h);
  option_ptr = uicontrol('visible','off','userdata',option);
  path_ptr = uicontrol('visible','off','userdata',[]);

  % SAVE WINDOW DATA AND LOCK
  H = nndArray({fig_axis desc_text surf_axis cont_axis surf_ptr cont_ptr ...
       var1 var2 var3 var4 radio1 radio2 radio3 option_ptr path_ptr});
  set(fig,'userdata',H,'nextplot','new')

  % INSTRUCTION TEXT
  feval(me,'instr');

  % LOCK WINDOW
  set(fig,'nextplot','new','color',nnltgray);

  nnchkfs;

%==================================================================
% Display the instructions.
%
% ME('instr')
%==================================================================

elseif strcmp(cmd,'instr') & ~isempty(fig)
  nnsettxt(desc_text,...
    'Use the radio buttons',...
    'to select the network',...
    'parameters to train',...
    'with backpropagation.',...
    '',...
    'The corresponding',...
    'error surface and',...
    'contour are shown',...
    'below.',...
    '',...
    'Click in the contour',...
    'graph (on the right)',...
    'to start the',...
    'steepest descent',...
    'learning algorithm.')
    
%==================================================================
% Respond to radio buttons.
%
% ME('radio',i)
%==================================================================

elseif strcmp(cmd,'radio') & ~isempty(fig) & (nargin == 2)

  % GET DATA
  option = get(option_ptr,'userdata');
  
  % ALTER TRAINABLE PARAMETERS
  if (arg1 ~= option)

    % HIGHLIGHT NEW RADIO BUTTON
    set(radios(option),'value',0)
    set(radios(arg1),'value',1)
    option = arg1;

    % CLEAR AXES
    delete(get(cont_axis,'children'))
    delete(get(surf_axis,'children'))

    % CONSTANTS
    W1 = [10; 10];
    b1 = [-5;5];
    W2 = [1 1];
    b2 = [-1];
    P = -2:0.1:2;
    [R,Q] = size(P);
    A1 = logsig(W1*P+b1*ones(1,Q));
    T = logsig(W2*A1+b2*ones(1,Q));
 
    % ERROR SURFACE & VARIABLE NAMES
    if option == 1
      load nndbp1
      set(variables([1 4]),'color',[1 1 1])
      set(variables([2 3]),'color',nndkblue)
    elseif option == 2
      load nndbp2
      set(variables([1 2]),'color',[1 1 1])
      set(variables([3 4]),'color',nndkblue)
    else
      load nndbp3
      set(variables([2 3]),'color',[1 1 1])
      set(variables([1 4]),'color',nndkblue)
    end

    set(fig,'nextplot','add')
    axes(cont_axis)
    set(get(cont_axis,'xlabel'),'string',v1)
    set(get(cont_axis,'ylabel'),'string',v2)
    set(cont_axis,'xlim',range1,'ylim',range2)
    [dummy,cont_h] = contour(x2,y2,E2,levels);
    set(cont_h,'CreateFcn','');
    plot3(range1([1 2 2 1 1]),range2([1 1 2 2 1]),1000*ones(1,5),...
      'color',nndkblue);
    cont_h2 = plot(optx,opty,'+','color',nnred);
    view(2)

    axes(surf_axis)
    set(get(surf_axis,'xlabel'),'string',v1)
    set(get(surf_axis,'ylabel'),'string',v2)
    set(surf_axis,'xlim',range1,'ylim',range2)
    surf_h = surf(x1,y1,E1);
    set(surf_h,'CreateFcn','','facecolor','interp')
    view(vw)

    % STORE DATA
    set(cont_ptr,'userdata',[cont_h; cont_h2]);
    set(path_ptr,'userdata',[]);
    set(option_ptr,'userdata',option);
  end

%==================================================================
% Respond to mouse down.
%
% ME('down')
%==================================================================

elseif strcmp(cmd,'down') & ~isempty(fig) & (nargin == 1)

  pt = get(cont_axis,'currentpoint');

  x = pt(1);
  y = pt(3);
  xlim = get(cont_axis,'xlim');
  ylim = get(cont_axis,'ylim');

  if (x > xlim(1) & x < xlim(2) & y > ylim(1) & y < ylim(2))

    % GET DATA
    option = get(option_ptr,'userdata');
    path = get(path_ptr,'userdata');
    cont_h = get(cont_ptr,'userdata');
    surf_h = get(surf_ptr,'userdata');

    % REMOVE PREVIOUS PATH
    set(fig,'nextplot','add')
    delete(path);

    % INITIAL VALUES
    W1 = [10; 10];
    b1 = [-5;5];
    W2 = [1 1];
    b2 = [-1];
    P = -2:0.1:2;
    [R,Q] = size(P);
    A1 = logsig(W1*P+b1*ones(1,Q));
    T = logsig(W2*A1+b2*ones(1,Q));

    % PLOT START POINT
    dkblue = nndkblue;
    red = nnred;
    axes(cont_axis);
    path = [...
      plot(x,y,'o','color',dkblue,'markersize',8,'CreateFcn','');
      plot(x,y,'o','color',[1 1 1],'markersize',10,'CreateFcn','');
      plot(x,y,'o','color',dkblue,'markersize',12,'CreateFcn','')];
    drawnow;

    % PLOT PATH
    set(fig,'pointer','watch')

    % INITIALIZE TRAINING
    if option == 1
      lr = 3.5;
      ep = 1000;
      W1(1,1) = x;
      W2(1,1) = y;
    elseif option == 2
      lr = 25;
      ep = 300;
      W1(1,1) = x;
      b1(1) = y;
    else
      lr = 25;
      ep = 60;
      b1(1) = x;
      b1(2) = y;
    end

    A1 = logsig(W1*P+b1*ones(1,Q));
    A2 = logsig(W2*A1+b2*ones(1,Q));
    E = T-A2;

    xx = [x zeros(1,ep)];
    yy = [y zeros(1,ep)];

    % TRAINING #1
    if option == 1
      for i=2:(ep+1)
        SSE = sumsqr(E);
        D2 = A2.*(1-A2).*E;
        D1 = A1.*(1-A1).*(W2'*D2);
        dW1 = D1*P'*lr;
        db1 = D1*ones(Q,1)*lr;
        dW2 = D2*A1'*lr;
        db2 = D2*ones(Q,1)*lr;
      
        newx = W1(1,1) + dW1(1,1); W1(1,1) = newx; xx(i) = newx;
        newy = W2(1,1) + dW2(1,1); W2(1,1) = newy; yy(i) = newy;

        A1 = logsig(W1*P+b1*ones(1,Q));
        A2 = logsig(W2*A1+b2*ones(1,Q));
        E = T-A2;
        ee(i) = sumsqr(E);
      end

    % TRAINING #2
    elseif option == 2
      for i=2:(ep+1)
        SSE = sumsqr(E);
        D2 = A2.*(1-A2).*E;
        D1 = A1.*(1-A1).*(W2'*D2);
        dW1 = D1*P'*lr;
        db1 = D1*ones(Q,1)*lr;
        dW2 = D2*A1'*lr;
        db2 = D2*ones(Q,1)*lr;
      
        newx = W1(1,1) + dW1(1,1); W1(1,1) = newx; xx(i) = newx;
        newy = b1(1)   + db1(1);   b1(1) = newy;   yy(i) = newy;

        A1 = logsig(W1*P+b1*ones(1,Q));
        A2 = logsig(W2*A1+b2*ones(1,Q));
        E = T-A2;
        ee(i) = sumsqr(E);
      end

   % TRAINING #3
   else
      for i=2:(ep+1)
        SSE = sumsqr(E);
        D2 = A2.*(1-A2).*E;
        D1 = A1.*(1-A1).*(W2'*D2);
        dW1 = D1*P'*lr;
        db1 = D1*ones(Q,1)*lr;
        dW2 = D2*A1'*lr;
        db2 = D2*ones(Q,1)*lr;
      
        newx = b1(1) + db1(1);   b1(1) = newx;   xx(i) = newx;
        newy = b1(2) + db1(2);   b1(2) = newy;   yy(i) = newy;

        A1 = logsig(W1*P+b1*ones(1,Q));
        A2 = logsig(W2*A1+b2*ones(1,Q));
        E = T-A2;
        ee(i) = sumsqr(E);
      end
    end

    % CONTOUR PLOT
    path = [path; plot(xx,yy,'color',nnred); plot(xx,yy,'o','color',nnred,'markersize',6)];
    set(fig,'nextplot','new')
    
    % SAVE DATA
    set(path_ptr,'userdata',path);
    set(fig,'pointer','arrow')

  end
end

