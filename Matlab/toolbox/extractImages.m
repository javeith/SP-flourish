clc, clear

%% Extract color/spectral images from rosbag

%% - Inputs:
imageBag = rosbag('/media/thanu/raghavshdd1/RealSenseXimeaCalib/Intel5/170810_185338/2017-08-10-18-53-38.bag');
specTop = select(imageBag,'Topic', '/ximea_asl/image_raw');
rgbTop = select(imageBag, 'Topic', '/falcon/realsense/color/image_raw');

% Extract every n-th image:
n = 1;

% RGB or spectral?
spectral = 1; % = [0, 1]

% Extract closest spectral image using header timestamps?
findClosest = 1; % = [0, 1]

% Specify folder for image output: 
saveLoc = '/media/thanu/raghavshdd1/RealSenseXimeaCalib/Intel5/170810_190723_pix4d/ximea_170810_190723';

%% Find image pairs
if findClosest == 1
    % Extract header timestamp
    for iImage = 1:rgbTop.NumMessages
        colorMsg = readMessages(rgbTop, iImage);
        colorTime(iImage) = colorMsg{1}.Header.Stamp.Sec + colorMsg{1}.Header.Stamp.Nsec / (10^9);
    end
    
    for iImage = 1:specTop.NumMessages
        specMsg = readMessages(specTop, iImage);
        specTime(iImage) = specMsg{1}.Header.Stamp.Sec + specMsg{1}.Header.Stamp.Nsec / (10^9); % REMOVE CONSTANT OFFSET!!!
    end
    
    % Find closest spectral image for each color image
    for iImage = 1:rgbTop.NumMessages
        %value to find
        val = colorTime(iImage);
        
        tmp = abs(specTime-val);
        
        %index of closest value
        [value, idx] = min(tmp);
        pairsColorXimea(iImage,:) = [iImage, idx, value];
    end
    
end

%% Extract
if spectral == 1 && findClosest == 0
    NoI = specTop.NumMessages;
elseif spectral == 1 && findClosest == 1
    NoI = size(pairsColorXimea,1);
else
    NoI = rgbTop.NumMessages;
end

for iImage = 1:n:NoI
    if spectral == 1 && findClosest == 0
        specImg = readSpecImage(specTop, iImage);
        image = specImg.dataCube(:,:,8);
    elseif spectral == 1 && findClosest == 1
        specImg = readSpecImage(specTop, pairsColorXimea(iImage,2));
        image = specImg.dataCube(:,:,8);
    else
        image = readImageROS(rgbTop, iImage);
    end
    
    if (iImage-1) < 10
        imwrite(image, [saveLoc '/frame000' num2str(iImage-1) '.jpg']);
        
    elseif (iImage-1) < 100
        imwrite(image, [saveLoc '/frame00' num2str(iImage-1) '.jpg']);
        
    elseif (iImage-1) < 1000
        imwrite(image, [saveLoc '/frame0' num2str(iImage-1) '.jpg']);
        
    elseif (iImage-1) < 10000
        imwrite(image, [saveLoc '/frame' num2str(iImage-1) '.jpg']);
        
    end
    
    % Display information
    if mod(iImage,10) == 0
        disp([num2str(iImage) ' out of ' num2str(NoI) ' images']);
    end
    
end