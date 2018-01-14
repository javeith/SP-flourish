
%    Created on: May 6, 2017
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
pcName = '/home/thanu/Documents/CWG-CALTag2/shortBag/pix4d/20170620-checkerboard-djix3-leej-173553_group1_densified_point_cloud.ply';

% - Frame parameters from Pix4D (_calibrated_camera_parameters.txt)
cameraParam = '/home/thanu/Documents/CWG-CALTag2/shortBag/pix4d/20170620-checkerboard-djix3-leej-173553_calibrated_camera_parameters.txt';

% - Intrinsic parameters for both cameras (.yaml)
cam0yaml = '/home/thanu/Documents/CWG-CALTag2/shortBag/DJIIntrinsics.yaml';
cam1yaml = '/home/thanu/Documents/CWG-CALTag2/shortBag/ximeaIntrinsics.yaml';

% - Extrinsic parameters (.info or .yaml)
exFile = '/home/thanu/Documents/CWG-CALTag2/shortBag/DJIXimeaExtrinsic.yaml';

% - Corresponding gimbal angles for each ximea image (atm: .mat; should be integrated in this script later)
load /home/thanu/Documents/CWG-CALTag2/shortBag/YPRXimea.mat;

% - cam1 image location
cam1Images = '/home/thanu/Documents/CWG-CALTag2/shortBag/ximea_173553/*.jpg';

% - Output location:
output = '/home/thanu/Documents/CWG-CALTag2/shortBag/ximeaPC.ply';

% - Sequoia? & If so, what channel?
sequoia = 0;        % [0, 1]
channel = 'red';    % [red, reg, nir, green]

% - Number of Batches:
NoB = 1000;

% - Downsample?
downSample = 0; % [0, 1]
gridStep = 0.03;

% -Remove frames?
removeFrames = 0;
frameStart = 7; % Number of frames to be removed at the start of frame list
frameEnd = 7; % Number of frames to be removed at the end of frame list

%% Read cam0 point cloud from Pix4D
cam0_PC = plyread(pcName);

if downSample == 1
    % Downsample
    cam0_PC = pcdownsample(cam0_PC,'gridAverage',gridStep);
    
    % Save point cloud
    pcwrite(cam0_PC,'downsampled_PC','PLYFormat','binary');
end

% Number of points in cloud
NoP = size(cam0_PC.Location,1);

% Init Color array (R,G,B)
cam0_PC.Color = uint8(zeros(NoP,3));

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

cameraPose = zeros(3,3,(size(w2cTrafo,1) / 5));

while w2cIt <= NoLextParam
    w2cTrafo(w2cIt,1) = extParam(w2cIt,1);
    w2cTrafo(w2cIt+1,:) = extParam(w2cIt+6,:);
    w2cTrafo(w2cIt+2,:) = extParam(w2cIt+7,:);
    w2cTrafo(w2cIt+3,:) = extParam(w2cIt+8,:);
    w2cTrafo(w2cIt+4,:) = extParam(w2cIt+9,:);
    w2cIt = w2cIt + 10;
end

w2cTrafo( ~any(w2cTrafo,2), : ) = [];

if removeFrames == 1
    w2cTrafo((end-frameEnd*5+1):end,:) = [];
    w2cTrafo(1:(frameStart*5),:) = [];
end

for poseIt = 1:5:size(w2cTrafo,1)
    cameraPose(:,:,poseIt) = w2cTrafo((poseIt+2):(poseIt+4),:);
end

cameraPose(:,:, any(~any(cameraPose,2))) = [];
cameraPose = num2cell(cameraPose,[1 2]);
cameraPose = blkdiag(cameraPose{:});

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
    T_10 = [R_10,r_10;0,0,0,1];
    
elseif extrinsicFormat == 1 && sequoia == 0
    extrinsic = ReadYaml(exFile);
    
    T_10 = cell2mat(extrinsic.cam1.T_cn_cnm1);
    
elseif extrinsicFormat == 1 && sequoia == 1
    extrinsic = ReadYaml(exFile);
    % Get green to rgb (color) matrix
    T_CG = cell2mat(extrinsic.rgb.T_cn_cnm1);
    
    % Get green to desired channel (spectral) matrix
    T_SG = cell2mat(eval(['extrinsic.',channel,'.T_cn_cnm1']));
    T_GC = [T_CG(1:3,1:3)', (-T_CG(1:3,1:3))'*T_CG(1:3,4);0,0,0,1];
    T_10 = T_SG*T_GC;
end


% Rotation matrices from gimbal angles and combine T_10 & T_gimbalAngles
srcFiles = dir(cam1Images);
NoF_1 = size(srcFiles,1);
T_gimbalAngles = zeros(4,4,NoF_1);
T_gimbalAngles(4,4,:) = 1;
T_10_eachFrame = T_gimbalAngles;

for i = 1:NoF_1   
    T_gimbalAngles(1:3,1:3,i) = eul2rotm([degtorad(YPRXimea(i,2)), degtorad(YPRXimea(i,3)), degtorad(YPRXimea(i,4))]);
    T_10_eachFrame(:,:,i) = T_gimbalAngles(:,:,i) * T_10;
end

NoF_0 = size(w2cTrafo,1) / 5;

clear yaw pitch roll gimbalPYRT correspondingGimbalAngles
clear q_x_cam0 q_x_cam1 q_y_cam0 q_y_cam1 q_z_cam0 q_z_cam1 q_w_cam0 q_w_cam1 C extrinsic close ...
    r_x_cam0 r_y_cam0 r_z_cam0 intrinsic0 intrinsic1 r_x_cam1 r_y_cam1 r_z_cam1 filename delimiter startRow ...
    formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers ...
    invalidThousandsSeparator thousandsRegExp me R extParam NoLextParam w2cIt cam0yaml cam1yaml ...
    cameraParam exFile pcName pathstr name ext distortionModel0 distortionModel1 extrinsicFormat T_gimbalAngles;

%% Remap 3D point from cam0 to pixel in cam1 + Extract color from cam1 images and save to cam0 images %%
% Read all cam1 images
srcFiles = dir(cam1Images);
imagePath = fileparts(cam1Images);

images = uint8(zeros(imageHeight1,imageWidth1,1,NoF_1)); % 3 channels ?!?!?


for ID = 1:NoF_1
    imageID = ID - 1;
    if sequoia == 1
        
        filename = strcat([imagePath '/'],srcFiles(imageID+1).name);
        image = imread(filename);
        
        % uint16 to uint8
        image = image./257;
        image = uint8(image);
        
        % convert to three channels
        %image = cat(3,image,image,image); % 1 or 3 channels ?!?!?!?!?!
        
        % Save to cell
        images(:,:,:,ID) = image;
        
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
        images(:,:,:,ID) = image;
%         if size(image,3) == 1
%             % convert to three channels
%             image = cat(3,image,image,image);
%             
%             % Save to matrix
%             images(:,:,:,ID) = image;
%         else
%             images(:,:,:,ID) = image;
%         end
    end
end

clear image srcFiles filename frameIt ID imageID poseIt imagePath;

%% Optimized reprojection
tic;

% Batch parameters
batchSize = ceil(NoP/NoB);
firstPoint = 1;

% Frame parameters
frameParam = 1:5:size(w2cTrafo,1);

% Rotation matrix from cam0 to cam1
rotationMatrix = T_10_eachFrame(1:3,1:3,:);
rotationMatrix = permute(rotationMatrix,[1 3 2]);
rotationMatrix = reshape(rotationMatrix,[NoF_1*3,3]);
rotationMatrix = kron(eye(NoF_0),rotationMatrix);

% Focal matrix from camera matrix (cam1)
focalMatrix = repmat(cameraMatrix1(1:2,1:2), 1, NoF_0);
focalMatrix = mat2cell(focalMatrix, 2, repmat(2,1,NoF_0));
focalMatrix = blkdiag(focalMatrix{:});

% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% focalMatrix = repmat(cameraMatrix1(1:2,1:2), 1, NoF_1*NoF_0);
% focalMatrix = mat2cell(focalMatrix, 2, repmat(2,1,NoF_1*NoF_0));
% focalMatrix = blkdiag(focalMatrix{:});
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

% Color matrix & Number of candidates per point
tempColor = zeros(NoP,1);  % 1 or 3 channels ?!?!?!?!?!
NoC = zeros(NoP,1);

for batchesIt = 1:NoB
    batchtime = tic;
    
    % Extract point coordinates from point cloud
    LocationMatrix = cam0_PC.Location(firstPoint:min((batchSize+firstPoint-1),end),:);
    NoP_batch = size(LocationMatrix,1);
    
    % Calculate distance from each point to each camera
    distance2Camera = bsxfun(@minus, LocationMatrix, reshape(w2cTrafo((frameParam+1),:).',[1,size(w2cTrafo((frameParam+1),:).')]));
    distance2Camera = permute(distance2Camera,[2,3,1]);
    distance2Camera = reshape(distance2Camera,[],NoP_batch);
    
    P_c1 = cameraPose * distance2Camera;
    
    clear distance2Camera LocationMatrix;
    
    t = reshape(T_10_eachFrame(1:3,4,:),[NoF_1*3,1]);
    t = repmat(t,NoF_0,1);
    P_c1 = rotationMatrix * P_c1 + t;
    
    % Add distortion model
    P_c1_1 = P_c1(1:3:end,:);
    P_c1_2 = P_c1(2:3:end,:);
    P_c1_3 = P_c1(3:3:end,:);
    
    clear P_c1;
    
    x_h = P_c1_1./P_c1_3;
    y_h = P_c1_2./P_c1_3;
    
    clear P_c1_3;
    
    r = (x_h).^2 + (y_h).^2;
    
    inFrame = find(r <= 1);
    
    p_hd_1 = zeros(size(r));
    p_hd_2 = zeros(size(r));
    
    if distortionModel == 0
        
        clear  P_c1_1 P_c1_2;
        
        theta = zeros(size(r));
        % Distortion from rovio.yaml file -> equidistant model
        % Formulas from
        % https://april.eecs.umich.edu/pdfs/richardson2013iros.pdf
        % (3)-(8)
        theta(inFrame) = atan(sqrt(r(inFrame)));
        theta(inFrame) = (theta(inFrame) + distortionCam1(1).*theta(inFrame) + distortionCam1(2).*theta(inFrame).^3 + distortionCam1(3).*theta(inFrame).^5 + distortionCam1(4).*theta(inFrame).^7);
        omega = atan2(y_h,x_h);
        p_hd_1(inFrame) = theta(inFrame).*cos(omega(inFrame));
        p_hd_2(inFrame) = theta(inFrame).*sin(omega(inFrame));
        
        clear theta omega r inFrame x_h y_h;
        
        p_hd = zeros(2*NoF_0*NoF_1, NoP_batch);
        p_hd(1:2:end,:) = p_hd_1;
        
        clear p_hd_1;
        
        p_hd(2:2:end,:) = p_hd_2;
        
        clear p_hd_2;
        
    elseif distortionModel == 1
        
        clear  P_c1_1 P_c1_2;
        
        % Distortion from rovio.yaml file -> RadTan model
        if NoCoeff == 4
            p_hd_1(inFrame) = (1 + distortionCam1(1).*r(inFrame) + distortionCam1(2).*r(inFrame).^2).*x_h(inFrame) + 2*distortionCam1(3).*x_h(inFrame).*y_h(inFrame) + distortionCam1(4)*(r(inFrame)+2*x_h(inFrame).^2);
            p_hd_2(inFrame) = (1 + distortionCam1(1).*r(inFrame) + distortionCam1(2).*r(inFrame).^2).*y_h(inFrame) + 2*distortionCam1(4).*x_h(inFrame).*y_h(inFrame) + distortionCam1(3)*(r(inFrame)+2*y_h(inFrame).^2);
            
        elseif NoCoeff == 5
            p_hd_1(inFrame) = (1 + distortionCam1(1).*r(inFrame) + distortionCam1(2).*r(inFrame).^2 + distortionCam1(3).*r(inFrame).^3).*x_h(inFrame) + 2*distortionCam1(4).*x_h(inFrame).*y_h(inFrame) + distortionCam1(5)*(r(inFrame)+2*x_h(inFrame).^2);
            p_hd_2(inFrame) = (1 + distortionCam1(1).*r(inFrame) + distortionCam1(2).*r(inFrame).^2 + distortionCam1(3).*r(inFrame).^3).*y_h(inFrame) + 2*distortionCam1(5).*x_h(inFrame).*y_h(inFrame) + distortionCam1(4)*(r(inFrame)+2*y_h(inFrame).^2);
            
        end
        
        clear r inFrame x_h y_h;
        
        p_hd = zeros(2*NoF_0*NoF_1, NoP_batch);
        p_hd(1:2:end,:) = p_hd_1;
        
        clear p_hd_1;
        
        p_hd(2:2:end,:) = p_hd_2;
        
        clear p_hd_2;
               
    elseif distortionModel == 2
        
        clear x_h y_h;
        
        % Fisheye for Sequoia
        % Equations from: https://tinyurl.com/y8ke9b3u
        theta = zeros(NoF_0,NoP_batch);
        theta(inFrame) = 2/pi * atan(sqrt(r(inFrame)));
        
        clear r;
        
        theta(inFrame) = (distortionCam1(2).*theta(inFrame) + distortionCam1(3).*theta(inFrame).^2 + distortionCam1(4).*theta(inFrame).^3 + distortionCam1(5).*theta(inFrame).^4);
        p_hd_1(inFrame) = theta(inFrame).*P_c1_1(inFrame) ./ sqrt(P_c1_1(inFrame).^2 + P_c1_2(inFrame).^2);
        p_hd_2(inFrame) = theta(inFrame).*P_c1_2(inFrame) ./ sqrt(P_c1_1(inFrame).^2 + P_c1_2(inFrame).^2);
        
        clear theta inFrame P_c1_1 P_c1_2;
        
        p_hd = zeros(2*NoF_0, NoP_batch);
        p_hd(1:2:end,:) = p_hd_1;
        
        clear p_hd_1;
        
        p_hd(2:2:end,:) = p_hd_2;
        
        clear p_hd_2;
    end
    
    clear inFrame omega theta r p_hd_1 p_hd_2 P_c1_1 P_c1_2 x_h y_h;
    
    % Calculate pixel coordinate using cameraMatrix
    p_hd(1:2:end) = p_hd(1:2:end) * cameraMatrix1(1,1);
    p_hd(2:2:end) = p_hd(2:2:end) * cameraMatrix1(2,2);
    p_c1 = p_hd + repmat(cameraMatrix1(1:2,3),NoF_0*NoF_1,1);
    
    clear p_hd;
    
    p_c1(p_c1 == cameraMatrix1(1,3)) = 0;
    p_c1(p_c1 == cameraMatrix1(2,3)) = 0;
    p_c1(p_c1 == 0) = NaN;
    
    p_c1_1 = p_c1(1:2:end,:);
    p_c1_2 = p_c1(2:2:end,:);
    
    clear p_c1;
    
    % Find the frames in which each point is present
    if sequoia == 1
        % For sequoia: Avoid pixels close to image borders due to bluriness
        % after distortion
        [frame,points] = find((imageWidth1/6) <= p_c1_1 & p_c1_1 <= (5*imageWidth1/6) & (imageHeight1/6) <= p_c1_2 & p_c1_2 <= (5*imageHeight1/6));
        index = find((imageWidth1/6) <= p_c1_1 & p_c1_1 <= (5*imageWidth1/6) & (imageHeight1/6) <= p_c1_2 & p_c1_2 <= (5*imageHeight1/6));
        
    else
        [frame,points] = find((imageWidth1/6) <= p_c1_1 & p_c1_1 <= (5*imageWidth1/6) & (imageHeight1/6) <= p_c1_2 & p_c1_2 <= (5*imageHeight1/6));
        frame = mod(frame,NoF_1);
        frame(frame == 0) = NoF_1;
        index = find(0 <= p_c1_1 & p_c1_1 <= imageWidth1 & 0 <= p_c1_2 & p_c1_2 <= imageHeight1);
    end
    
    points = points + firstPoint - 1;
    
    % Number of candidates for each point
    NoC(1:max(points)) = NoC(1:max(points)) + accumarray(points(:),1);
    
    if sequoia == 1
        % Handle points which do not have candidates yet:
        % Find closest candidate to the middle of the image
        if firstPoint == 1
            noCandInd = find(NoC((firstPoint):max(points)) == 0);
        else
            noCandInd = find(NoC((firstPoint+1):max(points)) == 0);
        end
        
        if noCandInd ~= 0
            p_c1_1_temp = abs(p_c1_1(:,noCandInd) - imageWidth1/2);
            p_c1_1_temp(p_c1_1(:,noCandInd) >= imageWidth1) = Inf;
            p_c1_1_temp(p_c1_1(:,noCandInd) <= 0) = Inf;
            p_c1_2_temp = abs(p_c1_2(:,noCandInd) - imageHeight1/2);
            p_c1_2_temp(p_c1_2(:,noCandInd) >= imageHeight1) = Inf;
            p_c1_2_temp(p_c1_2(:,noCandInd) <= 0) = Inf;
            
            % Find closest by using nanmin
            % [min2,frame2] = nanmin(p_c1_1_temp + p_c1_2_temp);
            % frame2 = frame2';
            
            % Find closest by using sort -> allows to choose more than one
            % closest candidate
            [sorted,frame2] = sort(p_c1_1_temp + p_c1_2_temp);
            % frame2 = frame2(1:2,:);
            % frame2 = frame2(:);
            % noCandInd = repelem(noCandInd,2);
            frame2 = frame2(1,:)';
            
            tempInd = sub2ind(size(p_c1_1),frame2,noCandInd);
            
            if firstPoint == 1
                points2 = noCandInd + firstPoint-1;
            else
                points2 = noCandInd + firstPoint;
            end
            
            NoC(1:max(points2)) = NoC(1:max(points2)) + accumarray(points2(:),1);
            
            index = [index;tempInd];
            index = sort(index);
            
            temp = [frame,points;frame2,points2];
            [Y,I] = sort(temp(:,2));
            temp = temp(I,:);
            frame = temp(:,1);
            points = temp(:,2);
            
            clear frame2 I min2 noCandInd p_c1_1_temp p_c1_2_temp points2 temp tempInd Y;
        end
    end
    
    % Extract pixel coordinates
    u = ceil(p_c1_1(index));
    v = ceil(p_c1_2(index));
    
    clear index p_c1_1 p_c1_2;
    
    % Get color from cam1 images
    totalPoints = size(points,1);
    
    for pointIt = 1:totalPoints
        tempColor(points(pointIt),1) = tempColor(points(pointIt),1) + double(images(v(pointIt),u(pointIt),1,frame(pointIt)));
        %tempColor(points(pointIt),2) = tempColor(points(pointIt),2) + double(images(v(pointIt),u(pointIt),2,frame(pointIt)));
        %tempColor(points(pointIt),3) = tempColor(points(pointIt),3) + double(images(v(pointIt),u(pointIt),3,frame(pointIt)));
    end
    
    firstPoint = batchSize * batchesIt + 1;
    
    batchTime = toc(batchtime)
    % Display information
    disp([num2str(batchesIt) ' out of ' num2str(NoB) ' batches']);
    
end

% Save color to point cloud
colorMatrix(:,1) = ceil(tempColor ./ NoC);
colorMatrix(:,2) = colorMatrix(:,1); % 1 or 3 channels ?!?!?!?!?!
colorMatrix(:,3) = colorMatrix(:,1); % 1 or 3 channels ?!?!?!?!?!

cam0_PC.Color = uint8(colorMatrix);
duration_in_s = toc;
% figure, pcshow(cam0_PC);

%% Save point cloud
pcwrite(cam0_PC,output);

clear colorMatrix frame i totalPoints NoC points tempColor u v batchesIt batchSize ...
    cameraPose firstPoint focalMatrix frameParam New_NoP pointIt rotationMatrix ...
    batchtime frame2;
