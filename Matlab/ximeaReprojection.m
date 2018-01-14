%% Reproject point cloud from all 25 bands
clc, clear

%% raghavshdd1 location
hddLoc = '/media/thanu/raghavshdd1/';

%% -Inputs:
pcName = [hddLoc 'thanujan/Datasets/FIP/20170622/testSet/NIR25/band8.ply'];

% - Frame parameters from Pix4D (_calibrated_camera_parameters.txt)
cameraParam = [hddLoc 'thanujan/Datasets/FIP/20170622/FIP_band8_170622_calibrated_camera_parameters.txt'];

% - Intrinsic parameters for both cameras (.yaml)
cam0yaml = [hddLoc 'thanujan/Datasets/FIP/20170622/intrinsics_ximea.yaml'];
cam1yaml = [hddLoc 'thanujan/Datasets/FIP/20170622/intrinsics_VIS16.yaml'];

% - Extrinsic parameters (.info or .yaml)
exFile = [hddLoc 'thanujan/Datasets/FIP/20170622/extrinsics_VIS16.yaml'];
% exFile = [hddLoc 'thanujan/Datasets/Ximea_Tamron/extrinsics_ximea.yaml';

% - cam1 image location (Folder needs all images!)
cam1Images = [hddLoc 'thanujan/Datasets/FIP/20170622/VIS_bands/'];

% - Output location:
output = [hddLoc 'thanujan/Datasets/FIP/20170622/testSet/VIS16/'];

% - NIR25 or VIS16?
type = 'VIS16';

% - Number of Batches:
NoB = 24;

% - Downsample?
downSample = 0; % [0, 1]
gridStep = 0.04;

%% Reproject
reprojectionXimea_func( pcName, cameraParam, cam0yaml, cam1yaml, exFile, ...
    cam1Images, output, 0, '', '', NoB, downSample, gridStep, ...
    1, type);
