clc, clear

%% Camera Calibration using Matlab vision toolbox functions
% - cam1 image location
camImages = '/home/thanu/Documents/calib_color/*.jpg';

%% Read all images
srcFiles = dir(camImages);
imagePath = fileparts(camImages);
images = cell(1,length(srcFiles));

for jpgIt = 1 : length(srcFiles)
    filename = strcat([imagePath '/'],srcFiles(jpgIt).name);
    images{jpgIt} = filename;
end

%% Detect checkerboard in the images
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(images);
imshow(images{1}); 

images = images(imagesUsed);
for i = 1:4
    I = imread(images{i});
    subplot(2, 2, i);
    imshow(I); hold on; plot(imagePoints(:,1,i), imagePoints(:,2,i), 'ro');
end

%% Generate checkerboard points
squareSize = 50;
[worldPoints] = generateCheckerboardPoints(boardSize,squareSize);

%% Estimate camera parameters
cameraParameters = estimateCameraParameters(imagePoints,worldPoints,'NumRadialDistortionCoefficients',3,'EstimateTangentialDistortion',true);

%% View undistorted image
cM = [cameraParameters.FocalLength(1), 0, cameraParameters.PrincipalPoint(1);
    0, cameraParameters.FocalLength(2), cameraParameters.PrincipalPoint(2);
    0, 0, 1];
dM = [cameraParameters.RadialDistortion(1); cameraParameters.RadialDistortion(2);...
    cameraParameters.RadialDistortion(3);...
    cameraParameters.TangentialDistortion(1); cameraParameters.TangentialDistortion(2)];

undistoredImages = undistort(imread(images{1}), cM, dM);
figure;
imshow(undistoredImages);