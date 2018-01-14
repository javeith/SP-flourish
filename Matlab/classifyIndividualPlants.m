
%    Created on: July 14, 2017
%    Author: Thanujan Mohanadasan
%    Institute: ETH Zurich, Autonomous Systems Lab

%% Classify individual plants: Extract individual plants & create a heightmap
clc, clear

%% raghavshdd1 location
hddLoc = '/media/thanu/raghavshdd1/';

%% Inputs:
% - Point cloud (.ply):
pcName = [hddLoc 'thanujan/Datasets/2017-05-18/DJI/20170518-sugarbeet-djix3-eschikon-n2-10m_group1_densified_point_cloud.ply'];

% - Intrinsic parameters (.yaml)
camYaml = '/home/thanu/Documents/SpatioTemporalSpectralMapping/spatio_temporal_spectral_mapping/Matlab/sensorsData/DJIX3/DJIIntrinsics.yaml';

% - Location to save heightmap image
heightmapName = [hddLoc 'thanujan/Datasets/2017-05-18/DJI/heightMapOrtho.png'];

%%
% - Individual plants are segmentable:
individualPlants = 1; % = 1: yes; = 0: no

% - Search radius -> plant size [m]
radiusPlant = 0.05;

% - Minimum points to be accepted as plant
minPoints = 5;

% - File needed to calculate GPS for each centroid:
% Frame parameters from Pix4D (_calibrated_camera_parameters.txt):
cameraParam = [hddLoc 'thanujan/Datasets/2017-05-18/DJI/20170518-sugarbeet-djix3-eschikon-n2-10m_calibrated_camera_parameters.txt'];

% - Sequoia? & If so, what channel?
sequoia = 0;        % [0, 1]
channel = 'red';    % [red, reg, nir, green]

% - Latitude, Longitude file from Pix4D for each camera (_calibrated_external_camera_parameters_wgs84.txt):
latlongFile = [hddLoc 'thanujan/Datasets/2017-05-18/DJI/20170518-sugarbeet-djix3-eschikon-n2-10m_calibrated_external_camera_parameters_wgs84.txt'];

%% Read point cloud from Pix4D
pCloud = plyread(pcName);

% Number of points in cloud
NoP = size(pCloud.Location,1);

clear pcName

%% T from external_parameters.txt (from Pix4D)
[ extParam ] = readExternalParametersFile( cameraParam, sequoia );

NoLextParam = size(extParam,1);
w2cIt = 1;
cameraT = zeros(NoLextParam,4);

while w2cIt <= NoLextParam
    cameraT(w2cIt,1) = extParam(w2cIt,1);
    cameraT(w2cIt,2:4) = extParam(w2cIt+6,:);
    
    w2cIt = w2cIt + 10;
end

cameraT( ~any(cameraT,2), : ) = [];

%% Get latitude & longitude for each camera
[ longlatAlt ] = readLatLongParametersFile( latlongFile );

clear formatSpec invalidThousandsSeparator latlongFile me NoLextParam numbers ...
    numericData R raw rawData regexstr result row startRow w2cIt ans cameraParam ...
    col dataArray delimiter extParam fileID

%% Camera matrix (intrinsics.yaml)
% Load camera matrix cam1
intrinsic = ReadYaml(camYaml);
cMatrix(1,:) = cell2mat(intrinsic.camera_matrix.data(1:3));
cMatrix(2,:) = cell2mat(intrinsic.camera_matrix.data(4:6));
cMatrix(3,:) = cell2mat(intrinsic.camera_matrix.data(7:9));

% Get image size
imageWidth = intrinsic.image_width;
imageHeight = intrinsic.image_height;

clear camyaml intrinsic

%% Segment green vertices from the remaining point cloud
% Compute Gitelson's vegetation index for each point
vegIndex = 2*double(pCloud.Color(:,2)) - double(pCloud.Color(:,1))...
    - double(pCloud.Color(:,3));

minInd = min(vegIndex);
rangeInd = range(vegIndex);

vegIndex = vegIndex - minInd;
vegIndex = vegIndex ./ rangeInd;

% Compute Otsu threshold
thresh = graythresh(vegIndex);

% Create green and soil point clouds
locationMatrix = pCloud.Location;
colorMatrix = pCloud.Color;
greenXYZ = locationMatrix(vegIndex >= thresh,:);
greenColor = colorMatrix(vegIndex >= thresh,:);
soilXYZ = locationMatrix(vegIndex < thresh,:);
soilColor = colorMatrix(vegIndex < thresh,:);

greenCloud = pointCloud(greenXYZ,'Color',greenColor);
soilCloud = pointCloud(soilXYZ,'Color',soilColor);

figure(1); pcshow(greenCloud);
figure(2); pcshow(soilCloud);
% pcwrite(soilCloud,'/media/thanu/raghavshdd1/2017-05-18_DJI/soil_cloud.ply');

clear soilColor colorMatrix minInd rangeInd soilXYZ thresh vegIndex

NoP = size(greenXYZ,1);

if individualPlants == 1
    %% Cluster extraction to segment individual plants
    % [class,type] = dbscan(greenXYZ, 8, 0.1);
    
    % [idx, C] = kmeans(greenXYZ(:,1:2),200);
    % clusterXYZ = greenXYZ(idx == 27,:);
    % clusterColor = greenColor(idx == 27,:);
    % clpc = pointCloud(clusterXYZ,'Color',clusterColor);
    % figure(5); pcshow(clpc);
    
    idx = rangesearch(greenXYZ, greenXYZ, radiusPlant);
    clusterID = 1;
    associated = [];
    
    for i = 1:NoP
        
        if ~ismember(i,associated)
            neighbors = cell2mat(idx(i));
            temp = [];
            j = 1;
            
            while j <= size(neighbors,1)
                temp = union(temp,cell2mat(idx(neighbors(j))),'stable');
                neighbors = union(neighbors,temp,'stable');
                j = j + 1;
            end
            
            clusters(clusterID,1) = mat2cell(sort(temp),[size(temp,1)]);
            associated = union(associated,temp);
            clusterID = clusterID +1;
        end
        
    end
    
    clear associated clusterID i idx j temp
    
    %% Plot clusters in different colors & Calculate centroid of each cluster
    figure(3); pcshow(greenCloud); hold on;
    clusterXYZ = [];
    clusterColor = [];
    centroid = zeros(size(clusters,1),3);
    
    for i = 1:size(clusters,1)
        
        if size(clusters{i},1) >= minPoints
            clusterXYZ = [clusterXYZ;greenXYZ(cell2mat(clusters(i)),:)];
            centroid(i,:) = mean(greenXYZ(cell2mat(clusters(i)),:));
            color = uint8(ceil(rand(1,3) * 255));
            clusterColor = [clusterColor;repmat(color,size(clusters{i},1),1)];
        end
        
    end
    
    clpc = pointCloud(clusterXYZ,'Color',clusterColor);
    pcshow(clpc,'MarkerSize',20);
    hold off;
    
    clear clusterXYZ clusterColor color
    
    %% Calculate GPS of each centroid
    wgs84 = wgs84Ellipsoid('meters');
    
    % Optimize
    err_func = @(d) getErr(cameraT,longlatAlt,d);
    options = optimoptions('fminunc','MaxFunctionEvaluations', 1000);
    [D,err] = fminunc(err_func, [0,0,0],options);
    
    lat0 = D(1);
    lon0 = D(2);
    h0 = D(3);
    
    centroidLongLatAlt = zeros(size(centroid));
    [centroidLongLatAlt(:,2),centroidLongLatAlt(:,1),centroidLongLatAlt(:,3)] = ned2geodetic(centroid(:,2),-centroid(:,1),-centroid(:,3),lat0,lon0,h0,wgs84);
    
    clear cameraT D F err err_func h0 lat0 lon0 i longlatAlt neighbors options wgs84
    
end

%% Rotate point cloud to be flat
% Principal Component Analysis
mainDir = pca(locationMatrix);

if ~isRigid(affine3d([mainDir,zeros(3,1);0,0,0,1]))
    eulAngles = rotm2eul(mainDir);
    mainDir = eul2rotm([0,-eulAngles(2),-eulAngles(3)]);
end

greenFlat = pctransform(greenCloud,affine3d([mainDir,zeros(3,1);0,0,0,1]));
% soilFlat = pctransform(soilCloud,affine3d([mainDir,zeros(3,1);0,0,0,1]));
% pCloud = pctransform(pCloud,affine3d([mainDir,zeros(3,1);0,0,0,1]));
% figure(4); pcshow(greenFlat);

clear mainDir

%% Plot heightmap of field
% Round height for subdivision of point cloud
heightVector = greenFlat.Location(:,3);

if individualPlants == 1
    NoP = size(clusters,1);
end

% Find max & min height in point cloud
maxHeight = max(heightVector);
minHeight = min(heightVector);

% Number of Spaces
NoS = 20;
heightRange = linspace(minHeight,maxHeight,NoS+1)';
color = jet(NoS);

heightMap = zeros(NoP,5);

if individualPlants == 1
    heightMap(:,1:2) = centroid(:,1:2);
else
    heightMap(:,1:2) = greenXYZ(:,1:2);
end

if individualPlants == 1
    
    for pointIt = 1:NoP
        averageHeight = mean(heightVector(clusters{pointIt}));
        
        for rangeIt = 1:NoS
            
            if averageHeight >= heightRange(rangeIt) && averageHeight < heightRange(rangeIt+1)
                heightMap(pointIt,3:5) = color(rangeIt,:);
            end
            
        end
        
    end
    
else
    
    for pointIt = 1:NoP
        
        for rangeIt = 1:NoS
            
            if heightVector(pointIt) >= heightRange(rangeIt) && heightVector(pointIt) < heightRange(rangeIt+1)
                heightMap(pointIt,3:5) = color(rangeIt,:);
            end
            
        end
        
    end
    
end

heightRange = heightRange - minHeight;
heightVector = heightVector - minHeight;

% Plot heightmap
figure(5)
set(gca,'fontsize',18)
hold on;
title('Plant Height [cm]')
grid on;
colormap('jet')
colorbar('TickLabels',round(heightRange*100,2))
scatter(heightMap(:,1),heightMap(:,2),15,heightMap(:,3:5),'filled')
hold off;

clear NoS maxHeight minHeight heightRange color averageHeight

%% Save height map as image for html (TBT)
% Initilize image size
point0 = cMatrix * [0;0;1];
point1 = cMatrix * [1;1;1];

% [m/pixel]
GSD_x = 1 / (point1(1) - point0(1));
GSD_y = 1 / (point1(2) - point0(2));

% Get max and min in x&y (point cloud)
max_x = max(locationMatrix(:,1));
min_x = min(locationMatrix(:,1));

max_y = max(locationMatrix(:,2));
min_y = min(locationMatrix(:,2));

range_x = max_x - min_x;
range_y = max_y - min_y;

% Calculate number of pixels of image
imageHeightMosaic = ceil(range_y / GSD_y);

% Scale
factor = ceil(imageHeightMosaic / imageHeight);
GSD_x = GSD_x*factor;
GSD_y = GSD_y*factor;

imageWidthMosaic = ceil(range_x / GSD_x);
imageHeightMosaic = ceil(range_y / GSD_y);

heightMapImage = uint8(zeros(imageHeightMosaic, imageWidthMosaic, 3));
regionSize = 21;

% Save color to image
for point = 1:size(heightMap,1)
    u = ceil((heightMap(point,1) + norm(min_x)) / GSD_x);
    v = ceil((max_y - heightMap(point,2)) / GSD_y);
    
    if u == 0
        u = 1;
    end
    
    if v == 0
        v = 1;
    end
    
    color = ceil(heightMap(point,3:5)*255);
    colorRegion = zeros(regionSize,regionSize,3);
    colorRegion(:,:,1) = repmat(color(1),[regionSize regionSize]);
    colorRegion(:,:,2) = repmat(color(2),[regionSize regionSize]);
    colorRegion(:,:,3) = repmat(color(3),[regionSize regionSize]);
    
    heightMapImage((v-(regionSize-1)/2):(v+(regionSize-1)/2),(u-(regionSize-1)/2):(u+(regionSize-1)/2),:) = colorRegion;
    
end

% Save heightmap image
imwrite(heightMapImage, heightmapName,'Transparency',[0 0 0]);

%% Functions
function [ err ] = getErr(cameraT,longlatAlt, D)
lat0 = D(1);
lon0 = D(2);
h0 = D(3);

wgs84 = wgs84Ellipsoid('meters');
[xNorth,yEast,zDown] = geodetic2ned(longlatAlt(:,2),longlatAlt(:,1),longlatAlt(:,3),lat0,lon0,h0,wgs84);

%Calculate error
errX = mean(xNorth - cameraT(:,2));
errY = mean(yEast - cameraT(:,3));
errZ = mean(zDown - cameraT(:,4));

err = sqrt(errX^2 + errY^2 + errZ^2);

end
