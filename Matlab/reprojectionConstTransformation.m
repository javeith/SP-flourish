
%    Created on: March 17, 2017
%    Author: Thanujan Mohanadasan
%    Institute: ETH Zurich, Autonomous Systems Lab

%% Associate color from cam1 to point cloud from cam0
% By reprojecting 3D coordinate in point cloud to pixel coordinate in image of
% cam0

% P: 3D coordinate [x;y;z]
% p: pixel [u;v]

clc, clear

%% v2: Candidates determination and color association combined to reduce calculation time
% Inputs needed:
% - Point cloud from Pix4D (.ply)
pcName = '/data/2017-04-07/color_group1_densified_point_cloud.ply';

% - Frame parameters from Pix4D (_calibrated_camera_parameters.txt)
cameraParam = '/data/2017-04-07/color_calibrated_camera_parameters.txt';

% - Intrinsic parameters for both cameras (.yaml)
cam0yaml = 'data/2017-04-07/color_falcon.yaml';
cam1yaml = 'data/2017-04-07/ximea_falcon.yaml';

% - Extrinsic parameters (.info or .yaml)
exFile = 'data/2017-04-07/extrinsics_falcon_ir_ximea.yaml';

% - cam1 image location
cam1Images = 'data/2017-04-07/ximea_Pix4D/*.jpg';

% - Sequoia? & If so, what channel?
sequoia = 0;        % [0, 1]
channel = 'nir';    % [red, reg, nir, green]

%% Init: Select Mode %%
% Averaging
% Mode = 'Average';

% Higher weight if flying lower; choose weight
% Mode = 'Height';
% height = 6;

% weight = 5;

% Only use candidates within a certain angle from the top view
Mode = 'Angle';
pixelRadius = 200;

% Combine Height % Angle
% Mode = 'HeightAngle';
% height = 6;
% pixelRadius = 200;

%% Read cam0 point cloud from Pix4D
% cam0_PC = plyread('data/flourish_group1_densified_point_cloud.ply');

cam0_PC = plyread(pcName);

% Number of points in cloud
NoP = size(cam0_PC.Location,1);

% Init Color array (R,G,B)
cam0_PC.Color = uint8(zeros(NoP,3));

% cam0_PC = pcdownsample(cam0_PC,'gridAverage',0.2);
%  Display cloud
% figure
% pcshow(cam0_PC);

%% Camera matrix (intrinsics.yaml)
% Load camera matrix cam0
intrinsic0 = ReadYaml(cam0yaml);
cameraMatrix0(1,:) = cell2mat(intrinsic0.camera_matrix.data(1:3));
cameraMatrix0(2,:) = cell2mat(intrinsic0.camera_matrix.data(4:6));
cameraMatrix0(3,:) = cell2mat(intrinsic0.camera_matrix.data(7:9));

% Load distortion coeffiecients cam1
% equidistant: [k1, k2, k3 ,k4] or radtan: [k1, k2, r1 ,r2]
distortionCam0 = cell2mat(intrinsic0.distortion_coefficients.data)';
distortionModel0 = intrinsic0.distortion_model;

% Load camera matrix cam1
intrinsic1 = ReadYaml(cam1yaml);
cameraMatrix1(1,:) = cell2mat(intrinsic1.camera_matrix.data(1:3));
cameraMatrix1(2,:) = cell2mat(intrinsic1.camera_matrix.data(4:6));
cameraMatrix1(3,:) = cell2mat(intrinsic1.camera_matrix.data(7:9));

% Load distortion coeffiecients cam1 [k1, k2, k3 ,k4]
% equidistant: [k1, k2, k3 ,k4] or radtan: [k1, k2, r1 ,r2]
distortionCam1 = cell2mat(intrinsic1.distortion_coefficients.data)';
distortionModel1 = intrinsic1.distortion_model;

if strcmp(distortionModel1, 'equidistant')
    distortionModel = 0;
elseif strcmp(distortionModel1, 'radtan')
    distortionModel = 1;
    NoCoeff = size(distortionCam1,1);
elseif strcmp(distortionModel1, 'fisheye')
    distortionModel = 2;
end

% Load image size
% imageWidth0 = intrinsic0.image_width;
% imageHeight0 = intrinsic0.image_height;
imageWidth1 = intrinsic1.image_width;
imageHeight1 = intrinsic1.image_height;

% Convert to radtan
% [rdis0(1), rdis0(2), tdis0(1), tdis0(2), rdis0(3)] = Equi2RadTan([imageWidth0,imageHeight0], distortionCam0, cameraMatrix0);
% [rdis1(1), rdis1(2), tdis1(1), tdis1(2), rdis1(3)] = Equi2RadTan([imageWidth1,imageHeight1], distortionCam1, cameraMatrix1);

%% Load extrinsics (.info) for transformation matrix
[pathstr,name,ext] = fileparts(exFile);

if strcmp(ext, '.info')
    extrinsicFormat = 0;
elseif strcmp(ext, '.yaml')
    extrinsicFormat = 1;
end

if extrinsicFormat == 0
    extrinsic = fopen(exFile);
    
    C = textscan(extrinsic, '%s','delimiter', '\n');
    
    q_im_cam0 = zeros(3,1);
    q_im_cam1 = zeros(3,1);
    r_cam0 = zeros(3,1);
    r_cam1 = zeros(3,1);
    
    % cam0
    q_x_cam0 = strsplit(C{1}{14});
    q_im_cam0(1) = str2double(cell2mat(q_x_cam0(2)));
    
    q_y_cam0 = strsplit(C{1}{15});
    q_im_cam0(2) = str2double(cell2mat(q_y_cam0(2)));
    
    q_z_cam0 = strsplit(C{1}{16});
    q_im_cam0(3) = str2double(cell2mat(q_z_cam0(2)));
    
    q_w_cam0 = strsplit(C{1}{17});
    q_real_cam0 = str2double(cell2mat(q_w_cam0(2)));
    
    r_x_cam0 = strsplit(C{1}{18});
    r_cam0(1) = str2double(cell2mat(r_x_cam0(2)));
    
    r_y_cam0 = strsplit(C{1}{19});
    r_cam0(2) = str2double(cell2mat(r_y_cam0(2)));
    
    r_z_cam0 = strsplit(C{1}{20});
    r_cam0(3) = str2double(cell2mat(r_z_cam0(2)));
    
    % cam1
    q_x_cam1 = strsplit(C{1}{26});
    q_im_cam1(1) = str2double(cell2mat(q_x_cam1(2)));
    
    q_y_cam1 = strsplit(C{1}{27});
    q_im_cam1(2) = str2double(cell2mat(q_y_cam1(2)));
    
    q_z_cam1 = strsplit(C{1}{28});
    q_im_cam1(3) = str2double(cell2mat(q_z_cam1(2)));
    
    q_w_cam1 = strsplit(C{1}{29});
    q_real_cam1 = str2double(cell2mat(q_w_cam1(2)));
    
    r_x_cam1 = strsplit(C{1}{30});
    r_cam1(1) = str2double(cell2mat(r_x_cam1(2)));
    
    r_y_cam1 = strsplit(C{1}{31});
    r_cam1(2) = str2double(cell2mat(r_y_cam1(2)));
    
    r_z_cam1 = strsplit(C{1}{32});
    r_cam1(3) = str2double(cell2mat(r_z_cam1(2)));
    
    fclose(extrinsic);
    
end

%% Get R & T from external_parameters.txt (from Pix4D)
delimiter = ' ';
startRow = 8;

formatSpec = '%s%s%s%[^\n\r]';

fileID = fopen(cameraParam,'r');

textscan(fileID, '%[^\n\r]', startRow-1, 'WhiteSpace', '', 'ReturnOnError', false, 'EndOfLine', '\r\n');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'ReturnOnError', false);

fclose(fileID);

% Replace non-numeric text with NaN.
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = dataArray{col};
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,2,3]
    % Converts text in the input cell array to numbers. Replaced non-numeric
    % text with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1)
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            if sequoia == 1 && mod(row,10) == 1
                result = sscanf(rawData{row},'IMG_%d_%d_%d');
                numbers = num2str(result(3));
            else
                result = regexp(rawData{row}, regexstr, 'names');
                numbers = result.numbers;
            end
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if any(numbers==',')
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'))
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric text to numbers.
            if ~invalidThousandsSeparator
                numbers = textscan(strrep(numbers, ',', ''), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch me
        end
    end
end

R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

extParam(:,1) = cell2mat(raw(:, 1));
extParam(:,2) = cell2mat(raw(:, 2));
extParam(:,3) = cell2mat(raw(:, 3));

NoLextParam = size(extParam,1);
w2cIt = 1;
w2cTrafo = zeros(NoLextParam,3);

while w2cIt <= NoLextParam
    w2cTrafo(w2cIt,1) = extParam(w2cIt,1);
    w2cTrafo(w2cIt+1,:) = extParam(w2cIt+6,:);
    w2cTrafo(w2cIt+2,:) = extParam(w2cIt+7,:);
    w2cTrafo(w2cIt+3,:) = extParam(w2cIt+8,:);
    w2cTrafo(w2cIt+4,:) = extParam(w2cIt+9,:);
    w2cIt = w2cIt + 10;
end

% cameraMatrix0(1,:) = extParam(2,:);
% cameraMatrix0(2,:) = extParam(3,:);
% cameraMatrix0(3,:) = extParam(4,:);

w2cTrafo( ~any(w2cTrafo,2), : ) = [];

%% Calculate transformation matrix between cameras, JPL for quaternions
if extrinsicFormat == 0
    % q_cam0 = [q_im_cam0;q_real_cam0];
    % q_cam1 = [q_im_cam1;q_real_cam1];
    
    % Quaternion to rotation matrix
    R_0I = (2*q_real_cam0^2-1)*eye(3) - 2*q_real_cam0*skew(q_im_cam0) + 2*(q_im_cam0*q_im_cam0');
    R_1I = (2*q_real_cam1^2-1)*eye(3) - 2*q_real_cam1*skew(q_im_cam1) + 2*(q_im_cam1*q_im_cam1');
    R_10 = R_1I * R_0I';
    
    % Distance between cameras
    r_10 = r_cam1 - r_cam0;
    
    % Transformation matrix
    T_10 = [R_10,r_10];
    
elseif extrinsicFormat == 1 && sequoia == 0
    extrinsic = ReadYaml(exFile);
    
    T_10 = cell2mat(extrinsic.cam1.T_cn_cnm1);
    
elseif extrinsicFormat == 1 && sequoia == 1
    extrinsic = ReadYaml(exFile);
    % Get green to rgb (color) matrix
    T_CG = cell2mat(extrinsic.rgb.T_cn_cnm1);
    
    % Get green to desired channel (spectral) matrix
    T_SG = cell2mat(eval(['extrinsic.',channel,'.T_cn_cnm1']));
    
    T_10 = T_SG/T_CG;
end

%% ------------------------------------------------------------------------
% load data/2017-04-07/ximea2ir_oriloc_new.mat
% T_10 = tMatrix;
% T_10(1:3,4) = T_10(1:3,4) ./ 1000;


%% ------------------------------------------------------------------------

clear q_x_cam0 q_x_cam1 q_y_cam0 q_y_cam1 q_z_cam0 q_z_cam1 q_w_cam0 q_w_cam1 C extrinsic close ...
    r_x_cam0 r_y_cam0 r_z_cam0 intrinsic0 intrinsic1 r_x_cam1 r_y_cam1 r_z_cam1 filename delimiter startRow ...
    formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers ...
    invalidThousandsSeparator thousandsRegExp me R extParam NoLextParam w2cIt cam0yaml cam1yaml ...
    cameraParam exFile pcName pathstr name ext distortionModel0 distortionModel1 extrinsicFormat;

%% Remap 3D point from cam0 to pixel in cam1 + Extract color from cam1 images and save to cam0 images %%
% Read all cam1 images
srcFiles = dir(cam1Images);
imagePath = fileparts(cam1Images);

if sequoia == 1
    NoCells = sscanf(srcFiles(length(srcFiles)).name,'IMG_%d_%d_%d');
    images = cell(1,NoCells(3));
    
else
    NoCells = sscanf(srcFiles(length(srcFiles)).name,'frame%d');
    images = cell(1,NoCells);
    
end

frameIt = 1;
while frameIt <= size(w2cTrafo,1)
    imageID = w2cTrafo(frameIt,1);
    
    if sequoia == 1
        filename = strcat([imagePath '/'],srcFiles(imageID+1).name);
        image = imread(filename);
        
        % uint16 to uint8
        image = image./257;
        image = uint8(image);
        
        % convert to three channels
        image = cat(3,image,image,image);
        
        % Save to cell
        images{imageID+1} = image;
    else
        
        if imageID < 10
            filename = [imagePath '/frame000' num2str(imageID) '.jpg'];
            
        elseif imageID < 100
            filename = [imagePath '/frame00' num2str(imageID) '.jpg'];
            
        elseif imageID < 1000
            filename = [imagePath '/frame0' num2str(imageID) '.jpg'];
            
        elseif imageID < 10000
            filename = [imagePath '/frame' num2str(imageID) '.jpg'];
            
        end
        
        image = imread(filename);
        
        if size(image,3) == 1
            % convert to three channels
            image = cat(3,image,image,image);
            
            % Save to cell
            images{imageID+1} = image;
        else
            images{imageID+1} = image;
        end
        
    end
    frameIt = frameIt + 5;
    
end

tic;
colorMatrix = uint8(zeros(NoP,3));
LocationMatrix = cam0_PC.Location;

for point = 1:NoP  % 1:NoP
    P_w0 = LocationMatrix(point,:)';
    
    frameIt = 1; % = 1 !!!!
    candIt = 1;
    
    % Maximum of 100 candidates assumed
    candidate = zeros(4,100);
    
    while frameIt <= size(w2cTrafo,1)
        R_w2c = w2cTrafo((frameIt+2):(frameIt+4),:);
        T_w2c = w2cTrafo((frameIt+1),:)';
        
        % 3D location in camera frame
        P_c0 = [0;0;0;1];
        P_w2c = (P_w0 - T_w2c);
        P_c0(1:3) = R_w2c * P_w2c;
        
        % Transform to cam1
        P_c1 = T_10 * P_c0;
        % P_c1 = P_c0;
        
        % Consider distortion
        x_h = P_c1(1)/P_c1(3);
        y_h = P_c1(2)/P_c1(3);
        r = x_h^2 + y_h^2;
        
        if r <= 1
            
            if distortionModel == 0
                % Distortion from rovio.yaml file -> equidistant model
                % Formulas from
                % https://april.eecs.umich.edu/pdfs/richardson2013iros.pdf
                % (3)-(8)
                theta = atan(sqrt(r));
                theta = (theta + distortionCam1(1)*theta + distortionCam1(2)*theta.^3 + distortionCam1(3)*theta.^5 + distortionCam1(4)*theta.^7);
                omega = atan2(y_h,x_h);
                x_hd = theta.*cos(omega);
                y_hd = theta.*sin(omega);
                
            elseif distortionModel == 1
                % Distortion from rovio.yaml file -> RadTan model
                if NoCoeff == 4
                    x_hd = (1 + distortionCam1(1)*r + distortionCam1(2)*r^2)*x_h + 2*distortionCam1(3)*x_h*y_h + distortionCam1(4)*(r+2*x_h^2);
                    y_hd = (1 + distortionCam1(1)*r + distortionCam1(2)*r^2)*y_h + 2*distortionCam1(4)*x_h*y_h + distortionCam1(3)*(r+2*y_h^2);
                    
                elseif NoCoeff == 5
                    x_hd = (1 + distortionCam1(1)*r + distortionCam1(2)*r^2 + distortionCam1(3)*r^3)*x_h + 2*distortionCam1(4)*x_h*y_h + distortionCam1(5)*(r+2*x_h^2);
                    y_hd = (1 + distortionCam1(1)*r + distortionCam1(2)*r^2 + distortionCam1(3)*r^3)*y_h + 2*distortionCam1(5)*x_h*y_h + distortionCam1(4)*(r+2*y_h^2);
                    
                end
                
            elseif distortionModel == 2
                % Fisheye for Sequoia
                theta = 2/pi * atan(sqrt(r));
                theta = (distortionCam1(2)*theta + distortionCam1(3)*theta + distortionCam1(4)*theta.^2 + distortionCam1(5)*theta.^3);
                x_hd = theta.*P_c1(1) ./ sqrt(P_c1(1).^2 + P_c1(2).^2);
                y_hd = theta.*P_c1(2) ./ sqrt(P_c1(1).^2 + P_c1(2).^2);
            end
            
            % Get pixel coordinates
            p_c1 = cameraMatrix1 * [x_hd;y_hd;1];
            
            % Check if pixel is part of image
            if 0 <= p_c1(1) && p_c1(1) <= imageWidth1 && 0 <= p_c1(2) && p_c1(2)<= imageHeight1
                candidate(1,candIt) = w2cTrafo(frameIt,1);
                candidate(2,candIt) = p_c1(1);
                candidate(3,candIt) = p_c1(2);
                candidate(4,candIt) = P_c1(3);
                
                candIt = candIt+1;
            end
            
        end
        
        frameIt = frameIt+5;
    end
    
    candidate(:,candIt:end) = [];
    
    % Number of candidates
    NoC = size(candidate,2);
    RGB = uint8(zeros(NoC,3));
    
    for candIt2 = 1:NoC
        % RGB(candIt2,:) = impixel(images{candidate(1,candIt2)+1},ceil(candidate(2,candIt2)),ceil(candidate(3,candIt2)));
        image = images{candidate(1,candIt2)+1};
        RGB(candIt2,:) = image(ceil(candidate(3,candIt2)),ceil(candidate(2,candIt2)),:);
    end
    
    % Save color to location
    if strcmp(Mode,'Average')
        colorMatrix(point,:) = mean(RGB,1);
        % RMS
        % colorMatrix(point,:) = sqrt(sum(double(RGB).^2) / NoC);
        
    elseif strcmp(Mode,'Height')
        
        %         if any(candidate(4,:) <= 4)
        %
        %             totalWeight = 0;
        %             weightedColor = zeros(1,3);
        %
        %             for candIt2 = 1:NoC
        %
        %                 % Check height
        %                 if candidate(4,candIt2) <= height
        %                     weightedColor = weightedColor + weight * double(RGB(candIt2,:));
        %                     totalWeight = totalWeight + weight;
        %                 else
        %                     weightedColor = weightedColor + 1 * double(RGB(candIt2,:));
        %                     totalWeight = totalWeight + 1;
        %                 end
        %
        %             end
        %
        %             colorMatrix(point,:) = weightedColor./totalWeight;
        
        if any(candidate(4,:) <= height)
            heightCandidate = find(candidate(4,:) <= height);
            heightColor = zeros(1,3);
            
            for candIt2 = 1:size(heightCandidate,2)
                heightColor = heightColor + double(RGB(heightCandidate(candIt2),:));
            end
            
            colorMatrix(point,:) = heightColor ./ size(heightCandidate,2);
            heightCandidate = 0;
            
        else
            colorMatrix(point,:) = mean(RGB,1);
            % RMS
            % colorMatrix(point,:) = sqrt(sum(double(RGB).^2) / NoC);
        end
        
    elseif strcmp(Mode,'Angle')
        angleParam = zeros(1,NoC);
        
        for candIt2 = 1:NoC
            angleParam(candIt2) = sqrt(norm(candidate(2,candIt2) - cameraMatrix1(1,3))^2 + ...
                norm(candidate(3,candIt2) - cameraMatrix1(2,3))^2) / pixelRadius;
        end
        
        if any(angleParam <= 1)
            angleCandidate = find(angleParam <= 1);
            angleColor = zeros(1,3);
            
            for candIt2 = 1:size(angleCandidate,2)
                angleColor = angleColor + double(RGB(angleCandidate(candIt2),:));
            end
            
            colorMatrix(point,:) = angleColor ./ size(angleCandidate,2);
            angleCandidate = 0;
            
        else
            colorMatrix(point,:) = mean(RGB,1);
            % RMS
            % colorMatrix(point,:) = sqrt(sum(double(RGB).^2) / NoC);
        end
        
    elseif strcmp(Mode,'HeightAngle')
        angleParam = zeros(1,NoC);
        
        for candIt2 = 1:NoC
            angleParam(candIt2) = sqrt(norm(candidate(2,candIt2) - cameraMatrix1(1,3))^2 + ...
                norm(candidate(3,candIt2) - cameraMatrix1(2,3))^2) / pixelRadius;
        end
        
        if any(angleParam <= 1 & candidate(4,:) <= height)
            angleCandidate = find(angleParam <= 1 & candidate(4,:) <= height);
            angleColor = zeros(1,3);
            
            for candIt2 = 1:size(angleCandidate,2)
                angleColor = angleColor + double(RGB(angleCandidate(candIt2),:));
            end
            
            colorMatrix(point,:) = angleColor ./ size(angleCandidate,2);
            angleCandidate = 0;
            
        else
            colorMatrix(point,:) = mean(RGB,1);
            % RMS
            % colorMatrix(point,:) = sqrt(sum(double(RGB).^2) / NoC);
        end
        
    end
    
    % Display information
    if mod(point,1000) == 0
        disp([num2str(point) ' out of ' num2str(NoP) ' points']);
    end
    
end

cam0_PC.Color = colorMatrix;
duration_in_s = toc;

disp('STATUS: Corresponding frames for each 3D point determined and color associated');

clear candIt candIt2 frameIt P_c0 P_c1 p_c1 P_w0 point srcFiles filename images...
    jpgIt NoC candidate heightCandidate heightColor x_h x_hd y_h y_hd weight...
    angleColor pixelRadius angleParam colorMatrix RGB R_w2c T_w2c P_w2c image...
    omega theta cam1Images imagePath NoCells

%% Save point cloud
pcwrite(cam0_PC,'colored_PC','PLYFormat','binary');