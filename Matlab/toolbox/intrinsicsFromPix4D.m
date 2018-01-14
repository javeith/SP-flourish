%% Get Intrinsics from camera_parameters.txt
clc, clear

%% Inputs
% - Frame parameters from Pix4D (_calibrated_camera_parameters.txt)
cameraParam = '/media/thanu/raghavshdd1/Ximea_Tamron/20170613/N2_band8_170613/1_initial/params/N2_band8_170613_calibrated_camera_parameters.txt';

% - Save location of .yaml file
saveLoc = '/media/thanu/raghavshdd1/Ximea_Tamron/20170613/N2_band8_170613/intrinsics_ximea.yaml';

%% Read intrinsics
delimiter = ' ';
startRow = 8;
endRow = 13;
formatSpec = '%s%s%s%[^\n\r]';

% Open the text file
fileID = fopen(cameraParam,'r');

% Read columns of data according to the format.
textscan(fileID, '%[^\n\r]', startRow-1, 'WhiteSpace', '', 'ReturnOnError', false);
dataArray = textscan(fileID, formatSpec, endRow-startRow+1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', ...
    true, 'ReturnOnError', false, 'EndOfLine', '\r\n');

% Close the text file
fclose(fileID);

% Extract information
temp = [dataArray{1:end-1}];
cameraMatrix = str2double(temp(2:4,1:3));
cameraMatrix = cameraMatrix';
cameraMatrix = cameraMatrix(:)';
temp = temp';
distortion = str2double(temp(13:end-1));
imageWidth = str2double(temp(2));
imageHeight = str2double(temp(3));

% Clear temporary variables
clearvars delimiter startRow endRow formatSpec fileID dataArray ans temp cameraParam;

%% Save to .yaml
% Create struct to be written
intrinsics.image_width = imageWidth;
intrinsics.image_height = imageHeight;
intrinsics.camera_name = 'cam0';
intrinsics.camera_matrix.rows = 3;
intrinsics.camera_matrix.cols = 3;
intrinsics.camera_matrix.data = cameraMatrix;
intrinsics.distortion_model = 'radtan';
intrinsics.distortion_coefficients.rows = 1;
intrinsics.distortion_coefficients.cols = 5;
intrinsics.distortion_coefficients.data = distortion;

% Write
WriteYaml(saveLoc, intrinsics);