%% Verify Random Forest with a different dataset
clc,clear

%% raghavshdd1 location
hddLoc = '/media/thanu/raghavshdd1/';

%% Inputs:
% TRAINING 01 - Mdl:
load([hddLoc 'thanujan/Datasets/xClassifier/x41bands/RandomForest/TREE.mat'])

% Verification set:
NIRLoc = [hddLoc 'thanujan/Datasets/FIP/20170622/testSet/NIR25/'];
VISLoc = [hddLoc 'thanujan/Datasets/FIP/20170622/testSet/VIS16/'];

% Save location for point cloud:
output = [hddLoc 'thanujan/Datasets/xClassifier/x41bands/RandomForest/testSet_FIP_170622_Tree.ply'];

%% Read point clouds & extract data for net
for iBand = 1:25
    pc = plyread([NIRLoc 'band'  num2str(iBand) '.ply']);
    testSet(iBand,:) = double(pc.Color(:,1));
    clear pc;
end

for iBand = 1:16
    pc = plyread([VISLoc 'band'  num2str(iBand) '.ply']);
    testSet(iBand+25,:) = double(pc.Color(:,1));
    clear pc;
end

%% Test the linear classifier
if size(testSet,2) > 10^6
    result1 = predict(Mdl,testSet(:,1:1000000)');
    result2 = predict(Mdl,testSet(:,1000001:end)');
    result = [result1;result2];
    clear result1 result2
else
    result = predict(Mdl,testSet');
end

%% Create point cloud with different color for each class v
colorMatrix = zeros(size(result,1),3);

% Soil
colorMatrix(strcmp(result,'soil'),:) = repmat([110,25,0],[size(colorMatrix(strcmp(result,'soil')),1),1]);
% Road
colorMatrix(strcmp(result,'road'),:) = repmat([255,128,0],[size(colorMatrix(strcmp(result,'road')),1),1]);
% buckWheat
colorMatrix(strcmp(result,'buckWheat'),:) = repmat([125,0,255],[size(colorMatrix(strcmp(result,'buckWheat')),1),1]);
% corn
colorMatrix(strcmp(result,'corn'),:) = repmat([0,0,255],[size(colorMatrix(strcmp(result,'corn')),1),1]);
% grass
colorMatrix(strcmp(result,'grass'),:) = repmat([255,255,0],[size(colorMatrix(strcmp(result,'grass')),1),1]);
% soyBean
colorMatrix(strcmp(result,'soyBean'),:) = repmat([0,137,255],[size(colorMatrix(strcmp(result,'soyBean')),1),1]);
% sugarBeet
colorMatrix(strcmp(result,'sugarBeet'),:) = repmat([255,0,0],[size(colorMatrix(strcmp(result,'sugarBeet')),1),1]);
% winterWheat
colorMatrix(strcmp(result,'winterWheat'),:) = repmat([0,255,0],[size(colorMatrix(strcmp(result,'winterWheat')),1),1]);

pc = plyread([NIRLoc 'band'  num2str(8) '.ply']);
resultCloud = pointCloud(pc.Location,'Color',uint8(colorMatrix));

% figure(2)
% pcshow(resultCloud)
pcwrite(resultCloud,output,'PLYFormat','binary');
