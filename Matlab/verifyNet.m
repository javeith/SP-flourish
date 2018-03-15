%% Verify net with a different dataset
clc,clear

%% raghavshdd1 location
hddLoc = '/Volumes/mac_jannic_2017/';

%% Inputs:
% TRAINING 01 - NET:
% load /media/thanu/raghavshdd1/Training01_170713_DJI_RGB/net.mat

% TRAINING 02 - NET:
% load /media/thanu/raghavshdd1/Training02_170630_DJI_RGB_height/net.mat

% TRAINING 03 - NET:
load([hddLoc 'thanujan/Datasets/xHealthClassifier/Training03_170630_DJI_RGB_height_Soil/net.mat'])


% Verification set - Healthy:
% healthyPCLoc = '/media/thanu/raghavshdd1/2017-07-13/DJI/healthy_green.ply';
% load '/media/thanu/raghavshdd1/Training02_170630_DJI_RGB_height/heightHealthy.mat'
pcLoc = [hddLoc 'thanujan/Datasets/2017-05-18/DJI/20170518-sugarbeet-djix3-eschikon-n2-10m_group1_densified_point_cloud.ply'];

% Save location for point cloud:
output = [hddLoc 'thanujan/Datasets/xHealthClassifier/Training03_170630_DJI_RGB_height_Soil/testSet_170518.ply'];

% "Continuous" coloring: Quantization steps
qS = 100000;

%% Read point clouds
% healthyPC = plyread(healthyPCLoc);
pC = plyread(pcLoc);

clear healthyPCLoc pcLoc

%% Extract data for net
testRGB = [double(pC.Color)';(pC.Location(:,3)-min(pC.Location(:,3)))'];
% t = repmat([0,1],size(testRGB,2),1)';

%% Test the classifier
result = net(testRGB);
resultIndices = vec2ind(result);

% figure(1)
% plotconfusion(t,result)

%% Create point cloud with contiuous colormap
if size(result,1) == 2
    % Healthy & unhealthy categories
    healthy = [0,255,0];
    unhealthy = [51,25,0];
    
    R = ceil(linspace(unhealthy(1),healthy(1),qS));
    G = ceil(linspace(unhealthy(2),healthy(2),qS));
    B = ceil(linspace(unhealthy(3),healthy(3),qS));
    
    colors = ceil(parula(qS)*255);
    
    % Color continuously: healthy plants green and rest brown (soil, unhealthy...)
    colorMatrix = zeros(size(result,2),3);
    
    for iCand = 1:size(result,2)
        health = ceil(result(2,iCand)*qS);
        %     colorMatrix(iCand,:) = [R(health),G(health),B(health)];
        colorMatrix(iCand,:) = colors(health,:);
    end
    
    % colorMatrix(find(resultIndices==2),:) = repmat(green,size(find(resultIndices==2),2),1);
    % colorMatrix(find(resultIndices==1),:) = repmat(brown,size(find(resultIndices==1),2),1);
    
    resultCloud = pointCloud(pC.Location,'Color',uint8(colorMatrix));
    
    figure(2)
    pcshow(resultCloud)
    %     pcwrite(resultCloud,output);
    
elseif size(result,1) == 3
    % With soil category
    healthy = [0,255,0];
    unhealthy = [255,255,0];
    soil = [51,25,0];
    
    RGB = [[ceil(linspace(soil(1),unhealthy(1),qS/2)),ceil(linspace(unhealthy(1),healthy(1),qS/2))]',...
        [ceil(linspace(soil(2),unhealthy(2),qS/2)),ceil(linspace(unhealthy(2),healthy(2),qS/2))]',...
        [ceil(linspace(soil(3),unhealthy(3),qS/2)),ceil(linspace(unhealthy(3),healthy(3),qS/2))]'];
    
    colorMatrix = zeros(size(result,2),3);
    
    for iCand = 1:size(result,2)
        health = ceil((result(3,iCand)-result(1,iCand)+1)/2*qS);
        if health == 0
            health = 1;
        end
        colorMatrix(iCand,:) = RGB(health,:);
    end
    
    resultCloud = pointCloud(pC.Location,'Color',uint8(colorMatrix));
    
    figure(2)
    pcshow(resultCloud)
    pcwrite(resultCloud,output);
end

%% Plot colorbar
hold on
axis off
colormap( RGB );
caxis([-1 1]);
set(gca,'fontsize',16,'FontWeight', 'bold')
h = colorbar('location','Southoutside',...
    'XTick',[-1 -0.5 0 0.5 1]);
hold off
