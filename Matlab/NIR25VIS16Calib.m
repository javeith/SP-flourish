%% NIR25 VIS16 Calibration
clc, clear
close all

%% raghavshdd1 location
hddLoc = '/media/thanu/raghavshdd1/';

%% -Inputs:
% - Point cloud from Pix4D (.ply)
pcName = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170613/N2_band8_170613/2_densification/point_cloud/calib_cloud.ply'];

% - Frame parameters from Pix4D (_calibrated_camera_parameters.txt)
cameraParam = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170613/N2_band8_170613/1_initial/params/N2_band8_170613_calibrated_camera_parameters.txt'];

% - Intrinsic parameters for both cameras (.yaml)
cam0yaml = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170613/N2_band8_170613/intrinsics_ximea.yaml'];
cam1yaml = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170613/N2_band8_170613/intrinsics_ximea.yaml'];

% - Extrinsic parameters (.info or .yaml)
exFile = [hddLoc 'thanujan/Datasets/Ximea_Tamron/extrinsics_ximea.yaml'];

% - Images for calibration
NIR25 = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170613/bands/band8/frame0168.tif'];
VIS16 = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170613/VIS_bands/band11/frame0168.tif'];

% - Sequoia? & If so, what channel?
sequoia = 0;        % [0, 1]
initial_channel = 'rgb'; % [rgb, red, reg, nir, green]
output_channel = 'nir';    % [rgb, red, reg, nir, green]

% - Number of Batches:
NoB = 1;

% - Remove frames?
removeFrames = 1;
frameStart = 130; % Number of frames to be removed at the start of frame list
frameEnd = 393-131; % Number of frames to be removed at the end of frame list

% - Is the radial distortion too strong?
strongDist = 1;

%% Read cam0 point cloud from Pix4D
cam0_PC = plyread(pcName);

% Number of points in cloud
NoP = size(cam0_PC.Location,1);

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

%% Remap 3D point from cam0 to pixel in cam1 + Extract color from cam1 images and save to cam0 images %%
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
    
    clear distance2Camera;
    
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
    pixel(1,:) = ceil(p_c1_1(index));
    pixel(2,:) = ceil(p_c1_2(index));
    
    clear index p_c1_1 p_c1_2;
    
    firstPoint = batchSize * batchesIt + 1;
    
    % Display information
    batchTime = toc(batchtime)
    disp([num2str(batchesIt) ' out of ' num2str(NoB) ' batches']);
    
end


%% Estimate f, c_x, c_y, R_21, t_21 --> 15 unknowns
figure(1); imshow(imread(NIR25));
figure(2); imshow(imread(VIS16));

% figure(3), pcshow(cam0_PC,  'MarkerSize' , 200);

% Determine corresponding 3D point to a NIR25 pixel
candidate = find(pixel(1,:) == 122 & pixel(2,:) == 170);
loc = LocationMatrix(points(candidate),:)

% Corresponding VIS16 pixel
imagePoints = [335,126; 335,139; 349,139; 349,126; 339,84; 171,250; 146,227];

% Corresponding 3D point coordinates
worldPoints = [14.773921,7.9673615,-66.725349; 15.656192,8.2135391,-66.659554; ...
    15.409385,9.0715904,-66.583878; 14.564192,8.8683233,-66.839821; ...
    11.792191,7.4091206,-66.824348; ...
    26.397545,-1.4394836,-66.444153; 25.231129,-3.6357617,-66.334846];

Origin_1 = w2cTrafo(2,:)';
cam_R_1 = w2cTrafo(3:5,:);

% Initial values
f_init = 900;
c_x_init = 250;
c_y_init = 120;
R_21_init = eye(3);
t_21_init = zeros(3,1);

x_0 = [f_init,c_x_init,c_y_init,R_21_init(:)',t_21_init(:)'];

options = optimoptions('fmincon', 'MaxFunctionEvaluations', 20000, ...
    'OptimalityTolerance', 1.0000e-20,'StepTolerance', 1.0000e-20, ...
    'MaxIterations',20000);

err_func_cP = @(x) cameraParamsOptProblem(x, cam_R_1, Origin_1, worldPoints, imagePoints);
con = @rotMatCon;

[VIS16_Params,err_Params] = fmincon(err_func_cP, x_0, [], [], [], [], [], [], con, options);
avg_err = sqrt(err_Params/size(imagePoints,1));

R_21 = [VIS16_Params(4:6);VIS16_Params(7:9);VIS16_Params(10:12)]
t_12 = VIS16_Params(13:15)
cM_VIS = [VIS16_Params(1),0,VIS16_Params(2);0,VIS16_Params(1),VIS16_Params(3);0,0,1]

% For finding incorrect pairs --> remove if error is high
for i = 1:size(imagePoints,1)
    temp = cM_VIS * (R_21 * ([cam_R_1, -cam_R_1 * Origin_1] * [worldPoints(i,:)';1]) + t_12);
    iP(i,:) = [temp(1)/temp(3),temp(2)/temp(3)];
end
error = iP - imagePoints;

% Some checks
% det(R_21)
% norm(R_21)
% rotm2eul(R_21)*180/pi
% norm(t_12)
