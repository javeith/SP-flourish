%% Create NDVI plot
clc, clear, close all

%% raghavshdd1 location
hddLoc = '/media/thanu/raghavshdd1/';

%% Inputs:

% - RGB orthomosaic
RGBortho = [hddLoc 'thanujan/Datasets/2017-05-18/Orthomosaics/CutRGB.png'];

% - Infrared orthomosaic
IRortho = [hddLoc 'thanujan/Datasets/2017-05-18/Orthomosaics/CutNIR.png'];

% - Output name:
output = [hddLoc 'thanujan/Datasets/2017-05-18/Orthomosaics/CutNDVI.png'];

% -------------------------------------------------------------------------

% - Define range to be considered:
rangeInit = 2; % [0, 2]; == 0: consider (maxValue - minValue) of NDVI

%% Read images:

RGB = im2double(imread(RGBortho));
IR = im2double(imread(IRortho));

clear RGBortho IRortho

%% Extract NearInfraRed & VISiblered

NIR = IR(:,:,1);
VIS = RGB(:,:,1);

%% Calculate NDVI

a = NIR - VIS;
b = NIR + VIS;

% a = 2*RGB(:,:,2)-RGB(:,:,1)-RGB(:,:,3);
% b= 2*(RGB(:,:,1)+RGB(:,:,2)+RGB(:,:,3));

NDVI = a ./ b;

NoE = numel(NDVI);

% for i = 1:NoE
%     if NDVI(i) < 0
%         NDVI(i) = -1 * NDVI(i);
%     end
% end

% NDVI(isnan(NDVI)) = 0;

%% Define colormap

n = 100;
R = zeros(1,n);
B = zeros(1,n);
G = zeros(1,n);
R(31:50) = linspace(0,1,20);
R(51:90) = linspace(0.8,0,40);
B(1:40) = linspace(1,0,40);
G(51:100) = linspace(1,1,50);
G(26:50) = linspace(0,1,25);


% colormap( [R(:), G(:), B(:)] );

% figure(1)
% surf(peaks)
% colorbar

%% Plot colorbar for mapbox
% fig1=figure;
% axis off
% colormap( [R(:), G(:), B(:)] );
% caxis([-1 1]);
% set(gca,'fontsize',16,'FontWeight', 'bold')
% h = colorbar('location','Southoutside',...
%   'XTick',[-1 -0.5 0 0.5 1]);

%% Add color to image according to NDVI index
rows = size(NDVI,1);
columns = size(NDVI,2);

imageNDVI = zeros(rows,columns,3);

% Get range
minIndex = min(min(NDVI));
maxIndex = max(max(NDVI));

if rangeInit == 0
    range = (maxIndex-minIndex) / n;
else
    range = rangeInit / n;
end

for i = 1:rows
    
    for j = 1:columns
        
        if isnan(NDVI(i,j))
            imageNDVI(i,j,:) = [0,0,0];
        else
            
            if rangeInit == 0
                slot = ceil((NDVI(i,j) - minIndex) / range);
            else
                slot = ceil((NDVI(i,j) + 1) / range);
            end
            
            if slot == 0
                slot = 1;
            end
            
            imageNDVI(i,j,:) = [R(slot), G(slot), B(slot)];
        end
    end
end

%% Save image
imwrite(imageNDVI, output,'Transparency',[0 0 0]);
figure
imshow(imageNDVI);