%% Verify Self-Organizing Map Neural Network
clear, clc

%% raghavshdd1 location
hddLoc = '/media/thanu/raghavshdd1/';

%% Inputs:
% Load network:
load([hddLoc 'thanujan/Datasets/xClassifier/x41bands/SOM/3x3/SoM_3x3.mat'])

% Test set:
NIRLoc = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170510/clouds/'];
VISLoc = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170510/VIS_clouds/'];

% Image input?
imageBool = 0;
imgLoc = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170622/bands/'];
imageName = 'frame0200.tif';

% Load colorMapSOM:
load([hddLoc 'thanujan/Datasets/xClassifier/x41bands/SOM/3x3/colorMapSOM_OWN.mat'])

% Save location for point cloud:
output = [hddLoc 'thanujan/Datasets/xClassifier/x41bands/SOM/3x3/SOM_3x3_testSet_170510_OWNCOLORS.ply'];

%% Read point clouds & extract data for net
if imageBool == 1
    % Images
    for iBand = 1:25
        pc = imread([imgLoc 'band'  num2str(iBand) '/' imageName]);
        x(iBand,:) = double(pc(:));
        clear pc;
    end
else
    % Clouds
    for iBand = 1:25
        pc = plyread([NIRLoc 'band'  num2str(iBand) '.ply']);
        x(iBand,:) = double(pc.Color(:,1));
        clear pc;
    end
    
    for iBand = 1:16
        pc = plyread([VISLoc 'band'  num2str(iBand) '.ply']);
        x(iBand+25,:) = double(pc.Color(:,1));
        clear pc;
    end
end


%% Class vectors
% Here the self-organizing map is used to compute the class vectors of
% each of the training inputs.

y = net(x);
cluster_index = vec2ind(y);

%% Colormap
% close all
% for iClass = 1:16
%     colorMapSOM(iClass,:) = ceil([rand*255, rand*255, rand*255]);
% end

%% Create colored point cloud
colorMatrix = zeros(size(cluster_index,2),3);

for iClass = 1:size(y,1)
    colorMatrix(cluster_index == iClass,:) = repmat(colorMapSOM(iClass,:),[size(colorMatrix(cluster_index == iClass),2),1]);
end

if imageBool == 1
    img =  reshape(colorMatrix,216,409,3);
    imwrite(uint8(img), output, 'png');
else
    pc = plyread([NIRLoc 'band'  num2str(8) '.ply']);
    resultCloud = pointCloud(pc.Location,'Color',uint8(colorMatrix));
    pcwrite(resultCloud,output,'PLYFormat','binary');
end
