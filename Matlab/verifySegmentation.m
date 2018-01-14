%% Verify net with a different dataset
clc,clear

%% raghavshdd1 location
hddLoc = '/media/thanu/raghavshdd1/';

%% Inputs:
% TRAINING 01 - NET:
load([hddLoc 'thanujan/Datasets/xClassifier/x41bands/PatternNet/net.mat'])

% Verification set:
NIRLoc = [hddLoc 'thanujan/Datasets/FIP/20170622/testSet/NIR25/'];
VISLoc = [hddLoc 'thanujan/Datasets/FIP/20170622/testSet/VIS16/'];

% Save location for point cloud:
output = [hddLoc 'thanujan/Datasets/xClassifier/x41bands/PatternNet/testSet_FIP_170622_PatternNet.ply'];

% Threshold for "other" class:
threshold = 0.5;

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

%% Test the classifier
result = net(testSet);
resultIndices = vec2ind(result);

%% Create point cloud with contiuous colormap
% classes = {'Soil', 'Road', 'Buckwheat', 'Corn', 'Grass', 'Soybean', 'Sugarbeet', 'Winter wheat'};
colorMatrix = zeros(size(resultIndices,2),3);

for iPoint = 1:size(result,2)
    if result(1,iPoint) > threshold
        colorMatrix(iPoint,:) = [110,25,0];
    elseif result(2,iPoint) > threshold
        colorMatrix(iPoint,:) = [255,128,0];
    elseif result(3,iPoint) > threshold
        colorMatrix(iPoint,:) = [125,0,255];
    elseif result(4,iPoint) > threshold
        colorMatrix(iPoint,:) = [0,0,255];
    elseif result(5,iPoint) > threshold
        colorMatrix(iPoint,:) = [255,255,0];
    elseif result(6,iPoint) > threshold
        colorMatrix(iPoint,:) = [0,137,255];
    elseif result(7,iPoint) > threshold
        colorMatrix(iPoint,:) = [255,0,0];
    elseif result(8,iPoint) > threshold
        colorMatrix(iPoint,:) = [0,255,0];
    else
        colorMatrix(iPoint,:) = [192,192,192];
    end
end
% % Soil
% colorMatrix(resultIndices == 1,:) = repmat([110,25,0],[size(colorMatrix(resultIndices == 1),2),1]);
% % Road
% colorMatrix(resultIndices == 2,:) = repmat([255,128,0],[size(colorMatrix(resultIndices == 2),2),1]);
% % Crop 1
% colorMatrix(resultIndices == 3,:) = repmat([51,51,255],[size(colorMatrix(resultIndices == 3),2),1]);
% % Crop 2
% colorMatrix(resultIndices == 4,:) = repmat([255,0,0],[size(colorMatrix(resultIndices == 4),2),1]);
% % Crop 3
% colorMatrix(resultIndices == 5,:) = repmat([0,204,0],[size(colorMatrix(resultIndices == 5),2),1]);

pc = plyread([NIRLoc 'band'  num2str(8) '.ply']);
resultCloud = pointCloud(pc.Location,'Color',uint8(colorMatrix));

% figure(2)
% pcshow(resultCloud)
pcwrite(resultCloud,output,'PLYFormat','binary');
