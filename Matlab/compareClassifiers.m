%% Compare segmentation algorithms
% To be compared: Patternnet, LDA, QDA, Random-Forest
clc, clear

%% raghavshdd1 location
hddLoc = '/media/thanu/raghavshdd1/';

%% Inputs:
% PATTERNNET:
load([hddLoc 'thanujan/Datasets/xClassifier/x41bands/PatternNet/net.mat'])

% LDA:
load([hddLoc 'thanujan/Datasets/xClassifier/x41bands/LDA/LDA.mat'])

% QDA:
load([hddLoc 'thanujan/Datasets/xClassifier/x41bands/QDA/QDA.mat'])

% TreeBagger:
load([hddLoc 'thanujan/Datasets/xClassifier/x41bands/RandomForest/TREE.mat'])

% Plot save location:
saveLoc = [hddLoc 'thanujan/Datasets/xClassifier/testSet/x41bands/'];

%% Read point clouds & extract data for net
for iBand = 1:25
    corn1_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/corn/NIR25/band'  num2str(iBand) '.ply']);
    corn(iBand,:) = corn1_pc.Color(:,1);
    clear corn1_pc;
    
    sugarBeet_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/sugarbeet/NIR25/band'  num2str(iBand) '.ply']);
    sugarBeet(iBand,:) = sugarBeet_pc.Color(:,1);
    clear sugarBeet_pc;
    
    winterWheat1_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/winterwheat/NIR25/band'  num2str(iBand) '.ply']);
    winterWheat(iBand,:) = winterWheat1_pc.Color(:,1);
    clear winterWheat1_pc;
    
    buckWheat_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/buckwheat/NIR25/band'  num2str(iBand) '.ply']);
    buckWheat(iBand,:) = buckWheat_pc.Color(:,1);
    clear buckWheat_pc;
    
    grass_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/grass/NIR25/band'  num2str(iBand) '.ply']);
    grass(iBand,:) = grass_pc.Color(:,1);
    clear grass_pc;
    
    road_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/road/NIR25/band'  num2str(iBand) '.ply']);
    road(iBand,:) = road_pc.Color(:,1);
    clear road_pc;
    
    soil_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/soil/NIR25/band'  num2str(iBand) '.ply']);
    soil(iBand,:) = soil_pc.Color(:,1);
    clear soil_pc;
    
    soy_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/soybean/NIR25/band'  num2str(iBand) '.ply']);
    soyBean(iBand,:) = soy_pc.Color(:,1);
    clear soy_pc;
    
end

for iBand = 1:16
    corn1_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/corn/VIS16/band'  num2str(iBand) '.ply']);
    corn(iBand+25,:) = corn1_pc.Color(:,1);
    clear corn1_pc;

    sugarBeet_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/sugarbeet/VIS16/band'  num2str(iBand) '.ply']);
    sugarBeet(iBand+25,:) = sugarBeet_pc.Color(:,1);
    clear sugarBeet_pc;

    winterWheat1_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/winterwheat/VIS16/band'  num2str(iBand) '.ply']);
    winterWheat(iBand+25,:) = winterWheat1_pc.Color(:,1);
    clear winterWheat1_pc;

    buckWheat_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/buckwheat/VIS16/band'  num2str(iBand) '.ply']);
    buckWheat(iBand+25,:) = buckWheat_pc.Color(:,1);
    clear buckWheat_pc;

    grass_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/grass/VIS16/band'  num2str(iBand) '.ply']);
    grass(iBand+25,:) = grass_pc.Color(:,1);
    clear grass_pc;

    road_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/road/VIS16/band'  num2str(iBand) '.ply']);
    road(iBand+25,:) = road_pc.Color(:,1);
    clear road_pc;

    soil_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/soil/VIS16/band'  num2str(iBand) '.ply']);
    soil(iBand+25,:) = soil_pc.Color(:,1);
    clear soil_pc;

    soy_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/testSet/soybean/VIS16/band'  num2str(iBand) '.ply']);
    soyBean(iBand+25,:) = soy_pc.Color(:,1);
    clear soy_pc;

end

%% Preparing the Data
testSet = double([soil, road, buckWheat, corn, grass, soyBean, sugarBeet, winterWheat]);
t = [repmat([1,0,0,0,0,0,0,0],size(soil,2),1);repmat([0,1,0,0,0,0,0,0],size(road,2),1);repmat([0,0,1,0,0,0,0,0],size(buckWheat,2),1); ...
    repmat([0,0,0,1,0,0,0,0],size(corn,2),1); repmat([0,0,0,0,1,0,0,0],size(grass,2),1); repmat([0,0,0,0,0,1,0,0],size(soyBean,2),1); ...
    repmat([0,0,0,0,0,0,1,0],size(sugarBeet,2),1); repmat([0,0,0,0,0,0,0,1],size(winterWheat,2),1)]';

size(testSet)
size(t)

%% Run classifiers
% PATTERNNET
resultPN = net(testSet);

% LDA
if size(testSet,2) > 10^6
    result1 = predict(MdlLinear,testSet(:,1:1000000)');
    result2 = predict(MdlLinear,testSet(:,1000001:end)');
    resultLDA_cell = [result1;result2];
    clear result1 result2
else
    resultLDA_cell = predict(MdlLinear,testSet');
end

resultLDA = zeros(8,size(resultLDA_cell,1));
resultLDA(:,strcmp(resultLDA_cell,'soil')) = repmat([1,0,0,0,0,0,0,0],[size(resultLDA(strcmp(resultLDA_cell,'soil')),1),1])';
resultLDA(:,strcmp(resultLDA_cell,'road')) = repmat([0,1,0,0,0,0,0,0],[size(resultLDA(strcmp(resultLDA_cell,'road')),1),1])';
resultLDA(:,strcmp(resultLDA_cell,'buckWheat')) = repmat([0,0,1,0,0,0,0,0],[size(resultLDA(strcmp(resultLDA_cell,'buckWheat')),1),1])';
resultLDA(:,strcmp(resultLDA_cell,'corn')) = repmat([0,0,0,1,0,0,0,0],[size(resultLDA(strcmp(resultLDA_cell,'corn')),1),1])';
resultLDA(:,strcmp(resultLDA_cell,'grass')) = repmat([0,0,0,0,1,0,0,0],[size(resultLDA(strcmp(resultLDA_cell,'grass')),1),1])';
resultLDA(:,strcmp(resultLDA_cell,'soyBean')) = repmat([0,0,0,0,0,1,0,0],[size(resultLDA(strcmp(resultLDA_cell,'soyBean')),1),1])';
resultLDA(:,strcmp(resultLDA_cell,'sugarBeet')) = repmat([0,0,0,0,0,0,1,0],[size(resultLDA(strcmp(resultLDA_cell,'sugarBeet')),1),1])';
resultLDA(:,strcmp(resultLDA_cell,'winterWheat')) = repmat([0,0,0,0,0,0,0,1],[size(resultLDA(strcmp(resultLDA_cell,'winterWheat')),1),1])';

clear resultLDA_cell

% QDA
if size(testSet,2) > 10^6
    result1 = predict(MdlQuadratic,testSet(:,1:1000000)');
    result2 = predict(MdlQuadratic,testSet(:,1000001:end)');
    resultQDA_cell = [result1;result2];
    clear result1 result2
else
    resultQDA_cell = predict(MdlQuadratic,testSet');
end

resultQDA = zeros(8,size(resultQDA_cell,1));
resultQDA(:,strcmp(resultQDA_cell,'soil')) = repmat([1,0,0,0,0,0,0,0],[size(resultQDA(strcmp(resultQDA_cell,'soil')),1),1])';
resultQDA(:,strcmp(resultQDA_cell,'road')) = repmat([0,1,0,0,0,0,0,0],[size(resultQDA(strcmp(resultQDA_cell,'road')),1),1])';
resultQDA(:,strcmp(resultQDA_cell,'buckWheat')) = repmat([0,0,1,0,0,0,0,0],[size(resultQDA(strcmp(resultQDA_cell,'buckWheat')),1),1])';
resultQDA(:,strcmp(resultQDA_cell,'corn')) = repmat([0,0,0,1,0,0,0,0],[size(resultQDA(strcmp(resultQDA_cell,'corn')),1),1])';
resultQDA(:,strcmp(resultQDA_cell,'grass')) = repmat([0,0,0,0,1,0,0,0],[size(resultQDA(strcmp(resultQDA_cell,'grass')),1),1])';
resultQDA(:,strcmp(resultQDA_cell,'soyBean')) = repmat([0,0,0,0,0,1,0,0],[size(resultQDA(strcmp(resultQDA_cell,'soyBean')),1),1])';
resultQDA(:,strcmp(resultQDA_cell,'sugarBeet')) = repmat([0,0,0,0,0,0,1,0],[size(resultQDA(strcmp(resultQDA_cell,'sugarBeet')),1),1])';
resultQDA(:,strcmp(resultQDA_cell,'winterWheat')) = repmat([0,0,0,0,0,0,0,1],[size(resultQDA(strcmp(resultQDA_cell,'winterWheat')),1),1])';

clear resultQDA_cell

% TreeBagger
if size(testSet,2) > 10^6
    result1 = predict(Mdl,testSet(:,1:1000000)');
    result2 = predict(Mdl,testSet(:,1000001:end)');
    resultTREE_cell = [result1;result2];
    clear result1 result2
else
    resultTREE_cell = predict(Mdl,testSet');
end

resultTREE = zeros(8,size(resultTREE_cell,1));
resultTREE(:,strcmp(resultTREE_cell,'soil')) = repmat([1,0,0,0,0,0,0,0],[size(resultTREE(strcmp(resultTREE_cell,'soil')),1),1])';
resultTREE(:,strcmp(resultTREE_cell,'road')) = repmat([0,1,0,0,0,0,0,0],[size(resultTREE(strcmp(resultTREE_cell,'road')),1),1])';
resultTREE(:,strcmp(resultTREE_cell,'buckWheat')) = repmat([0,0,1,0,0,0,0,0],[size(resultTREE(strcmp(resultTREE_cell,'buckWheat')),1),1])';
resultTREE(:,strcmp(resultTREE_cell,'corn')) = repmat([0,0,0,1,0,0,0,0],[size(resultTREE(strcmp(resultTREE_cell,'corn')),1),1])';
resultTREE(:,strcmp(resultTREE_cell,'grass')) = repmat([0,0,0,0,1,0,0,0],[size(resultTREE(strcmp(resultTREE_cell,'grass')),1),1])';
resultTREE(:,strcmp(resultTREE_cell,'soyBean')) = repmat([0,0,0,0,0,1,0,0],[size(resultTREE(strcmp(resultTREE_cell,'soyBean')),1),1])';
resultTREE(:,strcmp(resultTREE_cell,'sugarBeet')) = repmat([0,0,0,0,0,0,1,0],[size(resultTREE(strcmp(resultTREE_cell,'sugarBeet')),1),1])';
resultTREE(:,strcmp(resultTREE_cell,'winterWheat')) = repmat([0,0,0,0,0,0,0,1],[size(resultTREE(strcmp(resultTREE_cell,'winterWheat')),1),1])';

clear resultTREE_cell

%% Plot Confusion
% PATTERNNET
figure('units','normalized','outerposition',[0 0 1 1])
plotconfusion(t,resultPN)
set(gca,'fontsize',18);
title('Confusion Matrix: Pattern Recognition Network')
saveas(gca, [saveLoc 'confusionPATTERNNET'], 'png');
saveas(gca, [saveLoc 'confusionPATTERNNET'], 'fig');

% LDA
figure('units','normalized','outerposition',[0 0 1 1])
plotconfusion(t,resultLDA)
set(gca,'fontsize',18);
title('Confusion Matrix: Linear Discriminant Analysis')
saveas(gca, [saveLoc 'confusionLDA'], 'png');
saveas(gca, [saveLoc 'confusionLDA'], 'fig');

% QDA
figure('units','normalized','outerposition',[0 0 1 1])
plotconfusion(t,resultQDA)
set(gca,'fontsize',18);
title('Confusion Matrix: Quadratic Discriminant Analysis')
saveas(gca, [saveLoc 'confusionQDA'], 'png');
saveas(gca, [saveLoc 'confusionQDA'], 'fig');

% TREE
figure('units','normalized','outerposition',[0 0 1 1])
plotconfusion(t,resultTREE)
set(gca,'fontsize',18);
title('Confusion Matrix: Random Forest')
saveas(gca, [saveLoc 'confusionTREE'], 'png');
saveas(gca, [saveLoc 'confusionTREE'], 'fig');


%% Precision-Recall curve
classes = {'Soil', 'Road', 'Buckwheat', 'Corn', 'Grass', 'Soybean', 'Sugarbeet', 'Winter wheat'};

%PATTERNNET
for i = 1:size(t,1)
    [PREC, TPR, FPR, THRESH] = prec_rec(resultPN(i,:),t(i,:),'holdFigure', 1, 'plotROC', 0, 'plotPR', 1, 'numThresh', 1000, 'plotBaseline', 0);
    f1Scores = 2*(PREC.*TPR)./(PREC+TPR);
    f1Score(i) = max(f1Scores);
end
mean(f1Score)
sqrt(var(f1Score))

set(gcf,'units','normalized','outerpos',[0 0 1 1.2]);
set(gca,'fontsize',18);
grid on
title('Precision-Recall Curve: Pattern Recognition Network')
legend(classes, 'Location','southwest');
saveas(gca, [saveLoc 'prCurvePN'], 'png');
saveas(gca, [saveLoc 'prCurvePN'], 'fig');
close all

%LDA
for i = 1:size(t,1)
    [PREC, TPR, FPR, THRESH] = prec_rec(resultLDA(i,:),t(i,:),'holdFigure', 1, 'plotROC', 0, 'plotPR', 1, 'numThresh', 10000000, 'plotBaseline', 0);
    f1Scores = 2*(PREC.*TPR)./(PREC+TPR);
    f1Score(i) = max(f1Scores);
end
mean(f1Score)
sqrt(var(f1Score))

set(gcf,'units','normalized','outerpos',[0 0 1 1.2]);
set(gca,'fontsize',18);
grid on
title('Precision-Recall Curve: Linear Discriminant Analysis')
legend(classes, 'Location','southwest');
saveas(gca, [saveLoc 'prCurveLDA'], 'png');
saveas(gca, [saveLoc 'prCurveLDA'], 'fig');
close all

%QDA
for i = 1:size(t,1)
    [PREC, TPR, FPR, THRESH] = prec_rec(resultQDA(i,:),t(i,:),'holdFigure', 1, 'plotROC', 0, 'plotPR', 1, 'numThresh', 10000000, 'plotBaseline', 0);
    f1Scores = 2*(PREC.*TPR)./(PREC+TPR);
    f1Score(i) = max(f1Scores);
end
mean(f1Score)
sqrt(var(f1Score))

set(gcf,'units','normalized','outerpos',[0 0 1 1.2]);
set(gca,'fontsize',18);
grid on
title('Precision-Recall Curve: Quadratic Discriminant Analysis')
legend(classes, 'Location','southwest');
saveas(gca, [saveLoc 'prCurveQDA'], 'png');
saveas(gca, [saveLoc 'prCurveQDA'], 'fig');
close all

%TREE
for i = 1:size(t,1)
    [PREC, TPR, FPR, THRESH] = prec_rec(resultTREE(i,:),t(i,:),'holdFigure', 1, 'plotROC', 0, 'plotPR', 1, 'numThresh', 10000000, 'plotBaseline', 0);
    f1Scores = 2*(PREC.*TPR)./(PREC+TPR);
    f1Score(i) = max(f1Scores);
end
mean(f1Score)
sqrt(var(f1Score))

set(gcf,'units','normalized','outerpos',[0 0 1 1.2]);
set(gca,'fontsize',18);
grid on
title('Precision-Recall Curve: Random Forest')
legend(classes, 'Location','southwest');
saveas(gca, [saveLoc 'prCurveTREE'], 'png');
saveas(gca, [saveLoc 'prCurveTREE'], 'fig');
close all
