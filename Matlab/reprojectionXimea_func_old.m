function [] = reprojectionXimea_func_old( pcName, cameraParam, cam0yaml, cam1yaml, exFile, ...
    cam1Images, output, sequoia, initial_channel, output_channel, NoB, downSample, gridStep, ...
    strongDist, type)
%% OLD: Images read in every iteration (each batch)
%% Associate color from cam1 to point cloud from cam0
% By reprojecting 3D coordinate in point cloud to pixel coordinate in image of
% cam0

% P: 3D coordinate [x;y;z]
% p: pixel [u;v]

%% v2: Candidates determination and color association combined to reduce calculation time
% Inputs needed:
% - Point cloud from Pix4D (.ply)
% pcName = '';

% - Frame parameters from Pix4D (_calibrated_camera_parameters.txt)
% cameraParam = '';

% - Intrinsic parameters for both cameras (.yaml)
% cam0yaml = '';
% cam1yaml = '';

% - Extrinsic parameters (.info or .yaml)
% exFile = '';

% - cam1 image location (Folder needs all images!)
% cam1Images = '';

% - Output location:
% output = '';

% - Sequoia? & If so, what channel?
% sequoia = 0;        % [0, 1]
% initial_channel = 'rgb'; % [rgb, red, reg, nir, green]
% output_channel = 'nir';    % [rgb, red, reg, nir, green]

% - Number of Batches:
% NoB = 2;

% - Downsample?
% downSample = 0; % [0, 1]
% gridStep = 0.04;

% - Remove frames?
removeFrames = 0;
frameStart = 100; % Number of frames to be removed at the start of frame list
frameEnd = 381-102; % Number of frames to be removed at the end of frame list

% - Is the radial distortion too strong?
% strongDist = 1;

%% Read cam0 point cloud from Pix4D
cam0_PC = plyread(pcName);

if downSample == 1
    % Downsample
    cam0_PC = pcdownsample(cam0_PC,'gridAverage',gridStep);
    
    %     % Save point cloud
    %     pcwrite(cam0_PC,'downsampled_PC','PLYFormat','binary');
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

%% Get R & T from external_parameters.txt (from Pix4D)
[ extParam ] = readExternalParametersFile( cameraParam, sequoia, initial_channel );

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

w2cTrafo( [0;~any(w2cTrafo(2:end,:),2)] == 1, : ) = [];

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
extrinsic = ReadYaml(exFile);

T_10 = cell2mat(extrinsic.cam1.T_cn_cnm1);

clear q_x_cam0 q_x_cam1 q_y_cam0 q_y_cam1 q_z_cam0 q_z_cam1 q_w_cam0 q_w_cam1 C extrinsic close ...
    r_x_cam0 r_y_cam0 r_z_cam0 intrinsic0 intrinsic1 r_x_cam1 r_y_cam1 r_z_cam1 filename delimiter startRow ...
    formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers ...
    invalidThousandsSeparator thousandsRegExp me R extParam NoLextParam w2cIt cam0yaml cam1yaml ...
    cameraParam exFile pcName pathstr name ext distortionModel0 distortionModel1 extrinsicFormat;

%% Remap 3D point from cam0 to pixel in cam1 + Extract color from cam1 images and save to cam0 images %%
if strcmp(type,'NIR25')
    bands = [1:7,9:25];
elseif strcmp(type,'VIS16')
    bands = [1:16];
end

%% Optimized reprojection
tic;

% Batch parameters
batchSize = ceil(NoP/NoB);
firstPoint = 1;

% Frame parameters
frameParam = 1:5:size(w2cTrafo,1);
NoF = size(w2cTrafo,1) / 5;

% Rotation matrix from cam0 to cam1
rotationMatrix = repmat(T_10(1:3,1:3), 1, NoF);
rotationMatrix = mat2cell(rotationMatrix, 3, repmat(3,1,NoF));
rotationMatrix = blkdiag(rotationMatrix{:});

% Focal matrix from camera matrix (cam1)
focalMatrix = repmat(cameraMatrix1(1:2,1:2), 1, NoF);
focalMatrix = mat2cell(focalMatrix, 2, repmat(2,1,NoF));
focalMatrix = blkdiag(focalMatrix{:});

% Color matrix & Number of candidates per point
if strcmp(type,'NIR25')
    tempColor = zeros(NoP,1,25);  % 1 or 3 channels ?!?!?!?!?!
elseif strcmp(type,'VIS16')
    tempColor = zeros(NoP,1,16);  % 1 or 3 channels ?!?!?!?!?!
end

NoC = zeros(NoP,1);

for batchesIt = 1:NoB
    batchtime = tic;
    % Extract point coordinates from point cloud
    LocationMatrix = cam0_PC.Location(firstPoint:min((batchSize+firstPoint-1),end),:);
    New_NoP = size(LocationMatrix,1);
    
    % Calculate distance from each point to each camera
    distance2Camera = bsxfun(@minus, LocationMatrix, reshape(w2cTrafo((frameParam+1),:).',[1,size(w2cTrafo((frameParam+1),:).')]));
    distance2Camera = permute(distance2Camera,[2,3,1]);
    distance2Camera = reshape(distance2Camera,[],New_NoP);
    
    P_c1 = cameraPose * distance2Camera;
    
    clear distance2Camera LocationMatrix;
    
    P_c1 = rotationMatrix * P_c1 + repmat(T_10(1:3,4),NoF,1);
    
    % Add distortion model
    P_c1_1 = P_c1(1:3:end,:);
    P_c1_2 = P_c1(2:3:end,:);
    P_c1_3 = P_c1(3:3:end,:);
    
    clear P_c1;
    
    x_h = P_c1_1./P_c1_3;
    y_h = P_c1_2./P_c1_3;
    
    clear P_c1_3;
    
    r = (x_h).^2 + (y_h).^2;
    
    if strongDist == 1
        inFrame = find(r <= 0.35);
    else
        inFrame = find(r <= 1);
    end
    
    p_hd_1 = zeros(NoF,New_NoP);
    p_hd_2 = zeros(NoF,New_NoP);
    
    if distortionModel == 0
        
        clear  P_c1_1 P_c1_2;
        
        theta = zeros(NoF,New_NoP);
        % Distortion from rovio.yaml file -> equidistant model
        % Formulas from
        % https://april.eecs.umich.edu/pdfs/richardson2013iros.pdf
        % (3)-(8)
        theta(inFrame) = atan(sqrt(r(inFrame)));
        theta(inFrame) = (theta(inFrame) + distortionCam1(1).*theta(inFrame).^3 + distortionCam1(2).*theta(inFrame).^5 + ...
            distortionCam1(3).*theta(inFrame).^7 + distortionCam1(4).*theta(inFrame).^9);
        omega = atan2(y_h,x_h);
        p_hd_1(inFrame) = theta(inFrame).*cos(omega(inFrame));
        p_hd_2(inFrame) = theta(inFrame).*sin(omega(inFrame));
        
        clear theta omega r inFrame x_h y_h;
        
        p_hd = zeros(2*NoF, New_NoP);
        p_hd(1:2:end,:) = p_hd_1;
        
        clear p_hd_1frame;
        
        p_hd(2:2:end,:) = p_hd_2;
        
        clear p_hd_2;
        
    elseif distortionModel == 1
        
        clear  P_c1_1 P_c1_2;
        
        % Distortion from rovio.yaml file -> RadTan model
        if NoCoeff == 4
            p_hd_1(inFrame) = (1 + distortionCam1(1).*r(inFrame) + distortionCam1(2).*r(inFrame).^2).*x_h(inFrame) + ...
                2*distortionCam1(3).*x_h(inFrame).*y_h(inFrame) + distortionCam1(4)*(r(inFrame)+2*x_h(inFrame).^2);
            
            p_hd_2(inFrame) = (1 + distortionCam1(1).*r(inFrame) + distortionCam1(2).*r(inFrame).^2).*y_h(inFrame) + ...
                2*distortionCam1(4).*x_h(inFrame).*y_h(inFrame) + distortionCam1(3)*(r(inFrame)+2*y_h(inFrame).^2);
            
        elseif NoCoeff == 5
            p_hd_1(inFrame) = (1 + distortionCam1(1).*r(inFrame) + distortionCam1(2).*r(inFrame).^2 + ...
                distortionCam1(3).*r(inFrame).^3).*x_h(inFrame) + 2*distortionCam1(4).*x_h(inFrame).*y_h(inFrame) + ...
                distortionCam1(5)*(r(inFrame)+2*x_h(inFrame).^2);
            
            p_hd_2(inFrame) = (1 + distortionCam1(1).*r(inFrame) + distortionCam1(2).*r(inFrame).^2 + ...
                distortionCam1(3).*r(inFrame).^3).*y_h(inFrame) + 2*distortionCam1(5).*x_h(inFrame).*y_h(inFrame) + ...
                distortionCam1(4)*(r(inFrame)+2*y_h(inFrame).^2);
            
        end
        
        clear r inFrame x_h y_h;
        
        p_hd = zeros(2*NoF, New_NoP);
        p_hd(1:2:end,:) = p_hd_1;
        
        clear p_hd_1;
        
        p_hd(2:2:end,:) = p_hd_2;
        
        clear p_hd_2;
        
    elseif distortionModel == 2
        
        clear x_h y_h;
        
        % Fisheye for Sequoia
        % Equations from: https://tinyurl.com/y8ke9b3u
        theta = zeros(NoF,New_NoP);
        theta(inFrame) = 2/pi * atan(sqrt(r(inFrame)));
        
        clear r;
        
        theta(inFrame) = (distortionCam1(2).*theta(inFrame) + distortionCam1(3).*theta(inFrame).^2 + ...
            distortionCam1(4).*theta(inFrame).^3 + distortionCam1(5).*theta(inFrame).^4);
        
        p_hd_1(inFrame) = theta(inFrame).*P_c1_1(inFrame) ./ sqrt(P_c1_1(inFrame).^2 + P_c1_2(inFrame).^2);
        p_hd_2(inFrame) = theta(inFrame).*P_c1_2(inFrame) ./ sqrt(P_c1_1(inFrame).^2 + P_c1_2(inFrame).^2);
        
        clear theta inFrame P_c1_1 P_c1_2;
        
        p_hd = zeros(2*NoF, New_NoP);
        p_hd(1:2:end,:) = p_hd_1;
        
        clear p_hd_1;
        
        p_hd(2:2:end,:) = p_hd_2;
        
        clear p_hd_2;
    end
    
    clear inFrame omega theta r p_hd_1 p_hd_2 P_c1_1 P_c1_2 x_h y_h;
    
    % Calculate pixel coordinate using cameraMatrix
    p_c1 = focalMatrix * p_hd + repmat(cameraMatrix1(1:2,3),NoF,1);
    
    clear p_hd;
    
    p_c1(p_c1 == cameraMatrix1(1,3)) = 0;
    p_c1(p_c1 == cameraMatrix1(2,3)) = 0;
    p_c1(p_c1 == 0) = NaN;
    
    p_c1_1 = p_c1(1:2:end,:);
    p_c1_2 = p_c1(2:2:end,:);
    
    clear p_c1;
    
    % Find the frames in which each point is present
    [frame,points] = find(0 <= p_c1_1 & p_c1_1 <= imageWidth1 & 0 <= p_c1_2 & p_c1_2 <= imageHeight1);
    index = find(0 <= p_c1_1 & p_c1_1 <= imageWidth1 & 0 <= p_c1_2 & p_c1_2 <= imageHeight1);
    
    points = points + firstPoint - 1;
    
    % Number of candidates for each point
    NoC(1:max(points)) = NoC(1:max(points)) + accumarray(points(:),1);
    
    % Extract pixel coordinates
    u = ceil(p_c1_1(index));
    v = ceil(p_c1_2(index));
    
    clear index p_c1_1 p_c1_2;
    
    % Get color from cam1 images
    totalPoints = size(points,1); % ...,1); !!!
    
    for iBand = bands
        imageLoc = [cam1Images 'band' num2str(iBand) '/*.tif'];
        
        % Read all cam1 images
        srcFiles = dir(imageLoc);
        [~,format]=strtok(srcFiles(1).name,'.');
        imagePath = fileparts(imageLoc);
        
        NoF = size(w2cTrafo,1) / 5;
        images = uint8(zeros(imageHeight1,imageWidth1,1,NoF));
        
        frameIt = 1;
        ID = 1;
        
        while frameIt <= size(w2cTrafo,1)
            imageID = w2cTrafo(frameIt,1);
            
            if imageID < 10
                filename = [imagePath '/frame000' num2str(imageID) format];
            elseif imageID < 100
                filename = [imagePath '/frame00' num2str(imageID) format];
            elseif imageID < 1000
                filename = [imagePath '/frame0' num2str(imageID) format];
            elseif imageID < 10000
                filename = [imagePath '/frame' num2str(imageID) format];
            end
            
            image = imread(filename,'tiff');
            
            % Save to matrix
            images(:,:,:,ID) = image;
            
            frameIt = frameIt + 5;
            ID = ID + 1;
        end
        
        clear image srcFiles filename frameIt ID imageID poseIt imagePath;
        
        for pointIt = 1:totalPoints
            tempColor(points(pointIt),1,iBand) = tempColor(points(pointIt),1,iBand) + double(images(v(pointIt),u(pointIt),1,frame(pointIt)));
        end
        
        clear images
    end
    
    firstPoint = batchSize * batchesIt + 1;
    
    % Display information
    batchTime = toc(batchtime)
    disp([num2str(batchesIt) ' out of ' num2str(NoB) ' batches']);
    
end

for iBand = bands
    % Save color to point cloud
    colorMatrix(:,1) = ceil(tempColor(:,:,iBand) ./ NoC);
    colorMatrix(:,2) = colorMatrix(:,1); % 1 or 3 channels ?!?!?!?!?!
    colorMatrix(:,3) = colorMatrix(:,1); % 1 or 3 channels ?!?!?!?!?!
    
    cam0_PC.Color = uint8(colorMatrix);
    
    % Save point cloud
    pcwrite(cam0_PC,[output 'band' num2str(iBand) '.ply'],'PLYFormat','binary');
    cam0_PC.Color = uint8(zeros(NoP,3));
end

end