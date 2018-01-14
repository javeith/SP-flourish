%% Create an ENVI data with RGB, IR & 25 bands orthomosaics
clc, clear
% DATA WITH MORE THAN 3 DIMENSIONS IS NOT SUPPORTED (NDVI) ?!? -> solve
%% - Inputs:

% - RGB orthomosaic:
rgbOrtho = imread('/data/Orthomosaic/RGB2017-04-07.bmp');

% - Infrared orthomosaic:
irOrtho = imread('/data/Orthomosaic/IR2017-04-07.bmp');

% - Orthomosaic for each band:
spectralOrtho = imread('/data/Orthomosaic/ximea2017-04-07.bmp');

% - ENVI filename
filename = 'orthoCollection';

%% Combine to multidimensional array
imageWidth = size(rgbOrtho,1);
imageHeight = size(rgbOrtho,2);

ortho = zeros(imageWidth, imageHeight, 3, 2);
ortho(:,:,:,1) = rgbOrtho;
ortho(:,:,:,2) = irOrtho;
% orthi(:,:,:,3) = spectralOrtho;


%% Write to ENVI file
enviwrite(ortho, enviinfo(ortho), strcat(filename, '.bsq'), strcat(filename, '.hdr'));
