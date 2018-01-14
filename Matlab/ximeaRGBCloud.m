%% Create RGB cloud from VIS16 bands
clc, clear

%% raghavshdd1 location
hddLoc = '/media/thanu/raghavshdd1/';

%% Inputs:
% VIS cloud location:
visLoc = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170622/VIS_clouds/'];

% Save location:
saveLoc = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170622/VIS_clouds/rgbCloud.ply'];

%% Read clouds & create RGB cloud
% Read 630 nm
pcName = [visLoc 'band16.ply'];
cam0_PC = plyread(pcName);

% Number of points in cloud
NoP = size(cam0_PC.Location,1);
rgbMatrix = uint8(zeros(NoP,3));

rgbMatrix(:,1) = cam0_PC.Color(:,1);

% Read 546 nm
pcName = [visLoc 'band8.ply'];
cam0_PC = plyread(pcName);

rgbMatrix(:,2) = cam0_PC.Color(:,1);

% Read 465 nm
pcName = [visLoc 'band1.ply'];
cam0_PC = plyread(pcName);

rgbMatrix(:,3) = cam0_PC.Color(:,1);

% Save color to cloud
cam0_PC.Color = rgbMatrix;

% Save point cloud
pcwrite(cam0_PC,saveLoc,'PLYFormat','binary');