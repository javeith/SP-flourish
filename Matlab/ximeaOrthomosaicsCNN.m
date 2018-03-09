%% Information
% 
% This function creates orthomosaics as .png pictures from pointclouds.
% The necessary formats for this function to work are as follows:
% 
%     - "hddLoc" sets the path for the harddrive
%     - "cloudLoc" sets the path of the folder containing all the data
%     - "dataLabels" are the names of the folders in cloudLoc, in this case
%     the label classes
%     - "intrinsicsIndex" shows which dataLabel belongs to which intrinsics
%     - "filterTypes" are subfolders in dataLabels. In those folders there 
%     are the point clouds
%     
% Each point cloud set needs to be called "band*.ply", where * is the band 
% index. In each dataLabel folder there needs to be two folders called 
% "NIR25_Orthomosaic" and "VIS16_Orthomosaic", this is where the orthomosaics
% will be stored.
% 

%% Create orthomosaics for all bands
clc, clear

%% raghavshdd1 location
hddLoc = '/Volumes/mac_jannic_2017/';

%% Inputs:
ximea = 1;

% Names of the folders with labeled data
dataLabels = {'soil', 'road', 'Buckwheat', 'Corn1', 'Corn2', 'Grass', ...
    'Soybean', 'Sugarbeet', 'Winterwheat1', 'Winterwheat2'};
intrinsicsIndex = [1 2 3 2 2 3 3 2 2 3];
filterTypes = {'NIR25', 'VIS16'};

% Clouds location:
cloudLoc = [hddLoc 'thanujan/Datasets/xClassifier/trainSet/'];

% Intrinsics for gsd calculation:
intrinsicsLocation = {[hddLoc 'thanujan/Datasets/Ximea_Tamron/20170510/intrinsics_'];...
    [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170622/intrinsics_'];...
    [hddLoc 'thanujan/Datasets/FIP/20170622/intrinsics_']};
intrinsicsAddOn = { 'ximea.yaml';'VIS16.yaml'};

% Output location:
outLoc = [hddLoc 'thanujan/Datasets/xClassifier/trainSet/'];


%% Call orthomosaic_func for each orthomosaic
for filterType = filterTypes
    bands = setAmountOfBands(filterType);
    for dataLabel = dataLabels
        intrinsicSourcePath = [ char(intrinsicsLocation( intrinsicsIndex(strcmp(dataLabels,dataLabel)) )) ...
            char(intrinsicsAddOn( strcmp(filterTypes,filterType) ))];
        
        for iBand = bands
            pcName = [cloudLoc char(dataLabel) '/' char(filterType) '/band' num2str(iBand) '.ply'];
            output = [outLoc char(dataLabel) '/' char(filterType) '_Orthomosaic/band' num2str(iBand) '.png'];

            orthomosaic_func(pcName, intrinsicSourcePath, ximea, output);
        end
    end
end

function bands = setAmountOfBands(sub_folder_counter)
    if strcmp( sub_folder_counter,'NIR25') 
        bands = 1:25;
    elseif strcmp(sub_folder_counter,'VIS16')
        bands = 1:16;
    end
end