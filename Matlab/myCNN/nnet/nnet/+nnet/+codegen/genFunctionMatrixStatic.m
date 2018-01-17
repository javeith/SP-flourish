function code = genFunctionMatrixStatic(net,name,filepath,sampleColumn)

% Copyright 2012-2015 The MathWorks, Inc.

code = [...
    functionInterface(net,name) ...
    functionSummary(name) ...
    {'%'} ...
    nnet.codegen.commentText(functionHelp(net,filepath,name,sampleColumn)) ...
    {''} ...
    {'%#ok<*RPMT0>'} ...
    {''} ...
    {'% ===== NEURAL NETWORK CONSTANTS ====='} ...
    {''} ...
    nnet.codegen.constantBlock(net) ...
    {''} ...
    {'% ===== SIMULATION ========'} ...
    {''} ...
    simulationBlock(net,sampleColumn) ...
    {'end'} ...
    {''} ...
    {'% ===== MODULE FUNCTIONS ========'} ...
    {''} ...
    nnet.codegen.allModuleFunctions(net) ...
    ];
end

function code = functionInterface(net,name)
code = {['function ' functionCall(net,name)]};
end

function code = functionCall(net,name)
import nnet.codegen.*;
inputs = commaList(numberedStrings('x',net.numInputs));
outputs = commaList(numberedStrings('y',net.numOutputs));
code = ['[' outputs '] = ' name '(' inputs ')'];
end

function code = functionSummary(name)
code = {['%' upper(name) ' neural network simulation function.']};
end

function code = functionHelp(net,filepath,name,sampleColumn)
import nnet.codegen.*;
code = combineTextBlocks({...
    sourceSummary ...
    syntaxHelp(net,name,sampleColumn)  ...
    });
end

function code = sourceSummary
code = {['Generated by Neural Network Toolbox function genFunction, ' datestr(datevec(now)) '.']};
end

function code = syntaxHelp(net,name,sampleColumn)
import nnet.codegen.*;
code = {};
output2layer = find(net.outputConnect);
inputs = commaList(numberedStrings('x',net.numInputs));
outputs = commaList(numberedStrings('y',net.numOutputs));
code{end+1} = [functionCall(net,name) ' takes these arguments:'];
if(sampleColumn)
    for i=1:net.numInputs
        code{end+1} = ['  ',inputs(i),' = ',num2str(net.inputs{i}.size),'xQ matrix, input #' num2str(i)];
    end
else
    for i=1:net.numInputs
        code{end+1} = ['  ',inputs(i),' = Qx',num2str(net.inputs{i}.size),' matrix, input #' num2str(i)];
    end
end
code{end+1} = 'and returns:';
if(sampleColumn)
    for i=1:net.numOutputs
        ii = output2layer(i);
        code{end+1} = ['  ',outputs(i),' = ',num2str(net.outputs{ii}.size),'xQ matrix, output #' num2str(i)];
    end
else
    for i=1:net.numOutputs
        ii = output2layer(i);
        code{end+1} = ['  ',outputs(i),' = Qx',num2str(net.outputs{ii}.size),' matrix, output #' num2str(i)];
    end
end
code{end+1} = 'where Q is the number of samples.';
end

function code = simulationBlock(net,sampleColumn)
import nnet.codegen.*;
if needQ(net)
    if (net.numInputs > 0)
        if (sampleColumn)
            Qexp = 'size(x1,2)';
        else
            Qexp = 'size(x1,1)';
        end
    else
        Qexp = '0';
    end
    qBlock = { ...
        '% Dimensions' ...
        ['Q = ' Qexp '; % samples'] ...
        };
else
    qBlock = {};
end
inputBlocks = cell(1,net.numInputs);
for i=1:net.numInputs
    inputBlocks{i} = simulateInput(net,i,sampleColumn);
end
layerBlocks = cell(1,net.numLayers);
layerOrder = nn.layer_order(net);
for i=1:net.numLayers
    layerBlocks{i} = simulateLayer(net,layerOrder(i));
end
outputBlocks = cell(1,net.numOutputs);
for i=1:net.numOutputs
    outputBlocks{i} = simulateOutput(net,i,sampleColumn);
end
code = combineTextBlocks([{qBlock} inputBlocks layerBlocks outputBlocks]);
end

function flag = needQ(net)
flag = false;
for i=1:net.numLayers
    if net.biasConnect(i)
        flag = true; return;
    end
    if all([net.inputConnect(i,:) net.layerConnect(i,:)] == false)
        flag = true; return;
    end
end
end

function code = simulateInput(net,i,sampleColumn)
import nnet.codegen.*;
code = {};
code{end+1} = ['% Input ',num2str(i)];
if (~sampleColumn)
    code{end+1} = ['x' num2str(i) ' = x' num2str(i) ''';'];
end
numFcns = numel(net.inputs{i}.processFcns);
if (numFcns == 0)
    code{end+1} = '% no processing';
else
    lastVar = ['x' num2str(i)];
    nextVar = ['xp' num2str(i)];
    for j=1:numFcns
        module = net.inputs{i}.processFcns{j};
        if ~isempty(getStructFieldsFromMFile(module,'apply','settings'))
            settings = inputSettingName(i,j);
        else
            settings = '[]';
        end
        code{end+1} = [nextVar ' = ' module '_apply(' lastVar ',' settings ');'];
        lastVar = nextVar;
    end
end
end

function code = simulateLayer(net,i)
import nnet.codegen.*;
code = {};
code{end+1} = ['% Layer ',num2str(i)];
zcount = 0;
terms = {};
isNetprod = strcmp(net.layers{i}.netInputFcn,'netprod');
% Bias
if net.biasConnect(i)
    terms{end+1} = ['repmat(','b' num2str(i),',1,Q)'];
end
% Input Weights
for j=1:net.numInputs
    if net.inputConnect(i,j)
        module = net.inputWeights{i,j}.weightFcn;
        var = ['IW' num2str(i) '_' num2str(j)];
        if isempty(net.inputs{j}.processFcns)
            input = ['x' num2str(j)];
        else
            input = ['xp' num2str(j)];
        end
        if strcmp(module,'dotprod')
            terms{end+1} = [var,'*',input];
            if isNetprod, terms{end} = ['(' terms{end} ')']; end
        else
            zcount = zcount + 1;
            code{end+1} = ['z',num2str(zcount),' = ',module,'_apply(',var,',',input,');'];
            terms{end+1} = ['z' num2str(zcount)];
        end
    end
end
% Layer Weights
for j=1:net.numLayers
    if net.layerConnect(i,j)
        module = net.layerWeights{i,j}.weightFcn;
        var = ['LW' num2str(i) '_' num2str(j)];
        if strcmp(module,'dotprod')
            terms{end+1} = [var,'*a',num2str(j)];
            if isNetprod, terms{end} = ['(' terms{end} ')']; end
        else
            zcount = zcount + 1;
            code{end+1} = ['z',num2str(zcount),' = ',module,'_apply(',var,',a',num2str(j),');'];
            terms{end+1} = ['z' num2str(zcount)];
        end
    end
end
% Net Input Function
module = net.layers{i}.netInputFcn;
S = num2str(net.layers{i}.size);
if strcmp(module,'netsum')
    if isempty(terms)
        nExpression = ['zeros(' S ',Q)'];
    else
        nExpression = operatorList(terms,' + ');
    end
elseif strcmp(module,'netprod')
    if isempty(terms)
        nExpression = ['ones(' S ',Q)'];
    else
        nExpression = operatorList(terms,' .* ');
    end
else
    nExpression = [module,'({',commaList(terms),'},' S ',Q)'];
end
% Transfer Function
module = net.layers{i}.transferFcn;
if strcmp(module,'purelin')
    code{end+1} = ['a',num2str(i),' = ' nExpression ';'];
else
    code{end+1} = ['a',num2str(i),' = ',module,'_apply(',nExpression,');'];
end
end

function code = simulateOutput(net,i,sampleColumn)
import nnet.codegen.*;
code = {};
output2layer = find(net.outputConnect);
ii = output2layer(i);
avar = ['a',num2str(ii)];
yvar = ['y' num2str(i)];
code{end+1} = ['% Output ',num2str(i)];
numFcns = numel(net.outputs{ii}.processFcns);
if (numFcns == 0)
    code{end+1} = [yvar ' = ' avar ';'];
else
    for j=numFcns:-1:1
        module = net.outputs{ii}.processFcns{j};
        if ~isempty(getStructFieldsFromMFile(module,'reverse','settings'))
            settings = outputSettingName(i,j);
        else
            settings = '[]';
        end
        code{end+1} = [yvar ' = ' module '_reverse(' avar ',' settings ');'];
        avar = yvar;
    end
end
if(~sampleColumn)
    code{end+1} = [yvar ' = ' yvar ''';'];
end
end
