%% Create orthomosaics for all bands
clc, clear

%% raghavshdd1 location
hddLoc = '/Volumes/mac_jannic_2017/';

%% Inputs:
% NIR25 or VIS16?
type = 'NIR25';

% Clouds location:
cloudLoc = [hddLoc 'thanujan/Datasets/FIP/20170622/testSet/NIR25/'];

% Intrinsics for gsd calculation:
camyaml = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170510/intrinsics_ximea.yaml'];

% Output location:
outLoc = [hddLoc 'thanujan/Datasets/FIP/20170622/testSet/Orthomosaics/'];

%%
if strcmp(type,'NIR25')
    bands = [1:25];
elseif strcmp(type,'VIS16')
    bands = [1:16];
end

for iBand = bands
    pcName = [cloudLoc 'band' num2str(iBand) '.ply'];
    ximea = 1;
    output = [outLoc 'band' num2str(iBand) '.png'];
    
    orthomosaic_func(pcName, camyaml, ximea, output);
end
