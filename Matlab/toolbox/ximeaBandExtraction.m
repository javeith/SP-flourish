clc, clear

%% Extract bands from ximea images saved in a folder

%% - Inputs:
% Image location:
NIRLoc = '/media/thanu/raghavshdd1/Ximea_Tamron/20170613/NIR25/*.tif';
VISLoc = '/media/thanu/raghavshdd1/Ximea_Tamron/20170613/VIS16/*.tif';

% Specify folder for image output:
saveLoc = '/media/thanu/raghavshdd1/Ximea_Tamron/20170613/VIS_bands/';

% NIR25 or VIS16?
imgLoc = VISLoc;
type = 'VIS16';

%% Check imagepairs
imagesDeleted = checkNIR25VIS16Pairs( NIRLoc, VISLoc );

%% Read images
imgFiles = dir(imgLoc);
[~,ndx] = natsortfiles({imgFiles.name});
imgFiles = imgFiles(ndx);
NoI = length(imgFiles);

if strcmp(type, 'NIR25')
    nBands = 25;
elseif strcmp(type, 'VIS16')
    nBands = 16;
end

for iBand = 1:nBands
    [~,~,~] = mkdir([saveLoc 'band' num2str(iBand)]);
end

for iImg = 1:NoI
    img = imread(strcat(imgFiles(iImg).folder,'/',imgFiles(iImg).name));
    
    if strcmp(type, 'NIR25')
        img = SpectralImage(img);
    elseif strcmp(type, 'VIS16')
        img = VISImage(img);
    end
    
    fileID = iImg-1;
    if (fileID) < 10
        fileName = ['frame000' num2str(fileID) '.tif'];
    elseif (fileID) < 100
        fileName = ['frame00' num2str(fileID) '.tif'];
    elseif (fileID) < 1000
        fileName = ['frame0' num2str(fileID) '.tif'];
    elseif (fileID) < 10000
        fileName = ['frame' num2str(fileID) '.tif'];
    end
    
    for iBand = 1:nBands
        image = img.dataCube(:,:,iBand);
        imwrite(image, [saveLoc 'band' num2str(iBand) '/' fileName]);
    end
    
    % Display information
    if mod(iImg,10) == 0
        disp([num2str(iImg) ' out of ' num2str(NoI) ' images']);
    end
end