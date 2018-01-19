%% Create orthomosaics for all bands
clc, clear

%% raghavshdd1 location
hddLoc = '/Volumes/mac_jannic_2017/';

%% Inputs:
% Names of the folders with labeled data
folder = {'soil', 'road', 'Buckwheat', 'Corn1', 'Corn2', 'Grass', ...
    'Soybean', 'Sugarbeet', 'Winterwheat1', 'Winterwheat2'};
sub_folder = {'NIR25', 'VIS16'};

% Clouds location:
cloudLoc = [hddLoc 'thanujan/Datasets/xClassifier/trainSet/'];

% Intrinsics for gsd calculation:
camyaml = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170510/intrinsics_ximea.yaml'];

% Output location:
outLoc = [hddLoc 'thanujan/Datasets/xClassifier/trainSet/'];

%% Create directories for output



%% Call orthomosaic_func for each orthomosaic
for sub_folder_counter = sub_folder
    if strcmp( sub_folder_counter,'NIR25') 
        bands = 1:25;
    elseif strcmp(sub_folder_counter,'VIS16')
        bands = 1:16;
    end
    for folder_counter = folder
        for iBand = bands
            pcName = [cloudLoc char(folder_counter) '/' char(sub_folder_counter) '/band' num2str(iBand) '.ply'];
            ximea = 1;
            output = [outLoc char(folder_counter) '/' char(sub_folder_counter) '_Orthomosaic/band' num2str(iBand) '.png'];

            orthomosaic_func(pcName, camyaml, ximea, output);
        end
    end
end