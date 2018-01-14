%% Healthy Plant Classification using Random-Forest
clc, clear
close all

%% raghavshdd1 location
hddLoc = '/media/thanu/raghavshdd1/';

%% Inputs:
% TRAINING 01:
for iBand = 1:25
    corn1_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Corn1/NIR25/crop1_1_band'  num2str(iBand) '.ply']);
    size1 = size(corn1_pc.Color,1);
    corn(iBand,1:size1) = corn1_pc.Color(:,1);
    clear corn1_pc;
    
    corn2_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Corn2/NIR25/crop1_2_band'  num2str(iBand) '.ply']);
    corn(iBand,(size1+1):(size1+size(corn2_pc.Color,1))) = corn2_pc.Color(:,1);
    clear corn2_pc size1;
    
    sugarBeet_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Sugarbeet/NIR25/crop2_1_band'  num2str(iBand) '.ply']);
    sugarBeet(iBand,:) = sugarBeet_pc.Color(:,1);
    clear sugarBeet_pc;
    
    winterWheat1_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Winterwheat1/NIR25/crop3_1_band'  num2str(iBand) '.ply']);
    size2 = size(winterWheat1_pc.Color,1);
    winterWheat(iBand,1:size2) = winterWheat1_pc.Color(:,1);
    clear winterWheat1_pc;
    
    winterWheat2_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Winterwheat2/NIR25/winterwheat2_band'  num2str(iBand) '.ply']);
    winterWheat(iBand,(size2+1):(size2+size(winterWheat2_pc.Color,1))) = winterWheat2_pc.Color(:,1);
    clear winterWheat2_pc size2;
    
    buckWheat_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Buckwheat/NIR25/buckwheat_band'  num2str(iBand) '.ply']);
    buckWheat(iBand,:) = buckWheat_pc.Color(:,1);
    clear buckWheat_pc;
    
    grass_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Grass/NIR25/grass_band'  num2str(iBand) '.ply']);
    grass(iBand,:) = grass_pc.Color(:,1);
    clear grass_pc;
    
    road_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/road/NIR25/road_band'  num2str(iBand) '.ply']);
    road(iBand,:) = road_pc.Color(:,1);
    clear road_pc;
    
    soil_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/soil/NIR25/soil_band'  num2str(iBand) '.ply']);
    soil(iBand,:) = soil_pc.Color(:,1);
    clear soil_pc;
    
    soy_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Soybean/NIR25/soybean_band'  num2str(iBand) '.ply']);
    soyBean(iBand,:) = soy_pc.Color(:,1);
    clear soy_pc;
    
end

for iBand = 1:16
    corn1_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Corn1/VIS16/band'  num2str(iBand) '.ply']);
    size1 = size(corn1_pc.Color,1);
    corn(iBand+25,1:size1) = corn1_pc.Color(:,1);
    clear corn1_pc;
    
    corn2_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Corn2/VIS16/band'  num2str(iBand) '.ply']);
    corn(iBand+25,(size1+1):(size1+size(corn2_pc.Color,1))) = corn2_pc.Color(:,1);
    clear corn2_pc size1;
    
    sugarBeet_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Sugarbeet/VIS16/band'  num2str(iBand) '.ply']);
    sugarBeet(iBand+25,:) = sugarBeet_pc.Color(:,1);
    clear sugarBeet_pc;
    
    winterWheat1_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Winterwheat1/VIS16/band'  num2str(iBand) '.ply']);
    size2 = size(winterWheat1_pc.Color,1);
    winterWheat(iBand+25,1:size2) = winterWheat1_pc.Color(:,1);
    clear winterWheat1_pc;
    
    winterWheat2_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Winterwheat2/VIS16/winterwheat2_band'  num2str(iBand) '.ply']);
    winterWheat(iBand+25,(size2+1):(size2+size(winterWheat2_pc.Color,1))) = winterWheat2_pc.Color(:,1);
    clear winterWheat2_pc size2;
    
    buckWheat_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Buckwheat/VIS16/buckwheat_band'  num2str(iBand) '.ply']);
    buckWheat(iBand+25,:) = buckWheat_pc.Color(:,1);
    clear buckWheat_pc;
    
    grass_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Grass/VIS16/grass_band'  num2str(iBand) '.ply']);
    grass(iBand+25,:) = grass_pc.Color(:,1);
    clear grass_pc;
    
    road_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/road/VIS16/road_band'  num2str(iBand) '.ply']);
    road(iBand+25,:) = road_pc.Color(:,1);
    clear road_pc;
    
    soil_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/soil/VIS16/soil_band'  num2str(iBand) '.ply']);
    soil(iBand+25,:) = soil_pc.Color(:,1);
    clear soil_pc;
    
    soy_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Soybean/VIS16/soybean_band'  num2str(iBand) '.ply']);
    soyBean(iBand+25,:) = soy_pc.Color(:,1);
    clear soy_pc;
    
end

clear iBand

%% Preparing the Data
% Each ith column of the input matrix will have 41 elements:
% [41 Bands]

% TRAINING 01:
x = double([soil, road, buckWheat, corn, grass, soyBean, sugarBeet, winterWheat])';
t = [repmat({'soil'},size(soil,2),1);repmat({'road'},size(road,2),1);repmat({'buckWheat'},size(buckWheat,2),1); ...
    repmat({'corn'},size(corn,2),1); repmat({'grass'},size(grass,2),1);repmat({'soyBean'},size(soyBean,2),1); ...
    repmat({'sugarBeet'},size(sugarBeet,2),1); repmat({'winterWheat'},size(winterWheat,2),1)];

%% TreeBagger
% Train an ensemble of bagged classification trees using the entire data
% set. Specify |50| weak learners.  Store which observations are out of bag
% for each tree.
rng(1); % For reproducibility
Mdl = TreeBagger(50,x(1:5:end,:),t(1:5:end),'OOBPrediction','On','Method','classification','NumPrint',1,'Prior','Uniform');

%% Plot tree
% Plot a graph of the first trained classification tree.
% view(Mdl.Trees{1},'Mode','graph')

%% Out-of-bag
% Plot the out-of-bag error over the number of grown classification trees.
figure;
set(gca,'fontsize',18)
hold on
grid on
oobErrorBaggedEnsemble = oobError(Mdl);
plot(oobErrorBaggedEnsemble,'LineWidth',2)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';
title('Out-of-bag error over the number of grown classification trees')