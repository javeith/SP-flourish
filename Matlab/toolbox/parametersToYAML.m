%% Save camera parameters to .yaml file
clc, clear

% TODO: create extrinsics file

%% Inputs
% - Intrinsics - Location of.mat file to be saved as .yaml
matInt = '/home/thanu/Documents/CWG-CALTag2/shortBag/DJIIntrinsics.mat';
nameOfFile = '/home/thanu/Documents/CWG-CALTag2/shortBag/DJIIntrinsics.yaml';

% - Extrinsic - Location of.mat file to be saved as .yaml
matExt = '';

% - Sample images to get image size(s)
sample1 = '/home/thanu/Documents/CWG-CALTag2/shortBag/DJI_173553/DJI_0438.JPG';
sample2 = '';

%% Load
% Check what parameters to save
if ~isempty(matInt)
    load(matInt)
    param = 0;          % Intrinsics
    image1 = imread(sample1);
    
elseif ~isempty(matExt)
    load(matExt)
    param = 1;          % Extrinsic
    image1 = imread(sample1);
    image2 = imread(sample2);

end

clear matExt matInt sample1 sample2

%% Write intrinsics to .yaml
cameraParams = DJIcameraParams;

if param ==  0
    % Create struct to be written
   intrinsics.image_width = size(image1,2);
   intrinsics.image_height = size(image1,1);
   intrinsics.camera_name = 'cam0';
   intrinsics.camera_matrix.rows = 3;
   intrinsics.camera_matrix.cols = 3;
   intrinsics.camera_matrix.data = [cameraParams.FocalLength(1), 0, cameraParams.PrincipalPoint(1), ...
    0, cameraParams.FocalLength(2), cameraParams.PrincipalPoint(2), ...
    0, 0, 1];
   intrinsics.distortion_model = 'radtan';
   intrinsics.distortion_coefficients.rows = 1;
   intrinsics.distortion_coefficients.cols = size(cameraParams.RadialDistortion,2)...
       + size(cameraParams.TangentialDistortion,2);
   intrinsics.distortion_coefficients.data = [cameraParams.RadialDistortion, ...
       cameraParams.TangentialDistortion];
   
   % Write
   WriteYaml(nameOfFile, intrinsics);
   
end