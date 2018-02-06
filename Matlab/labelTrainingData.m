%
% Jannic Veith
% Autonomous Systems Lab
% ETH Zürich
%
%% Method description
%
% This script labels data with labels 0 to 8. For each desired label class
% the handle assignLabel below should be set to true. When set to true, an
% image will open with the expected class label in the terminal. In this
% image one will have to select a minimum of 3 points counter-clockwise, so
% that they form a convex polygon. Finish selection with enter, remove last
% point with backspace. All the pixels in this area with NDVI > cutoffNDVI 
% will be set to the corresponding class label.
%
% Input orthomosaic paths are pathBand1 and pathBand8.
%
% The labeled image will be saved to the path savePath as a .mat file.
% 

%% Set options
clc
clear all;
close all;

% Plots on or off
plotOn = true;

% Paths for orthomosaics for Band1 (700nm) and Band8 (803nm)
pathBand1 = '/Volumes/mac_jannic_2017/thanujan/Datasets/Ximea_Tamron/20170622/Orthomosaics/Band1.png';
pathBand8 = '/Volumes/mac_jannic_2017/thanujan/Datasets/Ximea_Tamron/20170622/Orthomosaics/Band8.png';

% Path to save the labeled data as .mat file
savePath = '/Users/jveith/Documents/2018 Semester Project/Python/trainingLabels.mat';

% Cutoff NDVI for best separation of plants and soil
cutoffNDVI = .51;

% Specify which plants appear in the image
assignCorn =        true;
assignSugarbeet =   true;
assignWinterwheat = true;
assignRoad =        false;
assignSoil =        false;
assignBuckwheat =   false;
assignGrass =       false;
assignSoybean =     false;


%% Labeling
% 
% Background, unlabeled 0
% Corn 1
% Sugarbeet 2
% Winterwheat 3
% Road 4
% Soil 5
% Buckwheat 6
% Grass 7
% Soybean 8

classNames = {'Background','Corn','Sugarbeet','Winterwheat','Road','Soil',...
    'Buckwheat','Grass','Soybean'};
classNumbers = {0, 1, 2, 3, 4, 5, 6, 7, 8};
classLabel = containers.Map(classNames,classNumbers);

%% Loading images and computing NDVI
orthomosaicBand1 = im2double(rgb2gray(imread(pathBand1)));
orthomosaicBand8 = im2double(rgb2gray(imread(pathBand8)));

% (Band8 - Band1) / (Band8 + Band1) = (803-700) / (803+700)
NDVI = (orthomosaicBand8 - orthomosaicBand1) ./ (orthomosaicBand1 + orthomosaicBand8);

% Threshholding 
CutNDVI=NDVI;
CutNDVI(NDVI>= cutoffNDVI) = 255;
CutNDVI(NDVI< cutoffNDVI) = 0;

%% Create labeling polygons via ordered edgepoints, counter clockwise

% Points [X,Y] are corners of a convex polygon, corner points must be
% ordered counter clockwise. Needs a minimum of 3 points.

% Ximea_Tamron/20170622
edgesCorn = [];
edgesSugarbeet = [];
edgesWinterwheat = [];
edgesRoad = [];
edgesSoil = [];
edgesBuckwheat = [];
edgesGrass = [];
edgesSoybean = [];

if assignCorn == true
    disp('Waiting for corner points for Corn...');
    [edgesCorn(:,1),edgesCorn(:,2),~] = impixel(CutNDVI);
    disp('Corn completed.');
end
if assignSugarbeet == true
    disp('Waiting for corner points for Sugarbeet...');
    [edgesSugarbeet(:,1),edgesSugarbeet(:,2),~] = impixel(CutNDVI);
    disp('Sugarbeet completed.');
end
if assignWinterwheat == true
    disp('Waiting for corner points for Winterwheat...');
    [edgesWinterwheat(:,1),edgesWinterwheat(:,2),~] = impixel(CutNDVI);
    disp('Winterwheat completed.');
end
if assignRoad == true
    disp('Waiting for corner points for Road...');
    [edgesRoad(:,1),edgesRoad(:,2),~] = impixel(CutNDVI);
    disp('Road completed.');
end
if assignSoil == true
    disp('Waiting for corner points for Soil...');
    [edgesSoil(:,1),edgesSoil(:,2),~] = impixel(CutNDVI);
    disp('Soil completed.');
end
if assignBuckwheat == true
    disp('Waiting for corner points for Buckwheat...');
    [edgesBuckwheat(:,1),edgesBuckwheat(:,2),~] = impixel(CutNDVI);
    disp('Buckwheat completed.');
end
if assignGrass == true
    disp('Waiting for corner points for Grass...');
    [edgesGrass(:,1),edgesGrass(:,2),~] = impixel(CutNDVI);
    disp('Grass completed.');
end
if assignSoybean == true
    disp('Waiting for corner points for Soybean...');
    [edgesSoybean(:,1),edgesSoybean(:,2),~] = impixel(CutNDVI);
    disp('Soybean completed.');
end

close figure(1)
%% Create orthomosaic containing numbers 0-8 as labels
labeledPicture = zeros(size(NDVI,1),size(NDVI,2),1);

for ii=1:size(labeledPicture,1)
    for jj=1:size(labeledPicture,2)
        if NDVI(ii,jj) >= cutoffNDVI 
            if isInsidePolygon([jj,ii],edgesCorn) == true
                labeledPicture(ii,jj)=classLabel('Corn');
            end
            if isInsidePolygon([jj,ii],edgesSugarbeet) == true
                labeledPicture(ii,jj)=classLabel('Sugarbeet');
            end
            if isInsidePolygon([jj,ii],edgesWinterwheat) == true
                labeledPicture(ii,jj)=classLabel('Winterwheat');
            end
            if isInsidePolygon([jj,ii],edgesRoad) == true
                labeledPicture(ii,jj)=classLabel('Road');
            end
            if isInsidePolygon([jj,ii],edgesSoil) == true
                labeledPicture(ii,jj)=classLabel('Soil');
            end
            if isInsidePolygon([jj,ii],edgesBuckwheat) == true
                labeledPicture(ii,jj)=classLabel('Buckwheat');
            end
            if isInsidePolygon([jj,ii],edgesGrass) == true
                labeledPicture(ii,jj)=classLabel('Grass');
            end
            if isInsidePolygon([jj,ii],edgesSoybean) == true
                labeledPicture(ii,jj)=classLabel('Soybean');
            end
        end
    end
end

save(savePath,'labeledPicture');

%% Create labeled color image

colors= [0,0,0;0,0,255;255,0,0;0,255,0;255,128,0;110,25,0;125,0,255;255,255,0;0,137,255];

colorRed = CutNDVI;
colorGreen= CutNDVI;
colorBlue = CutNDVI;
for ii= 1:8
    colorRed(labeledPicture==ii) = colors(ii+1,1);
    colorGreen(labeledPicture==ii) = colors(ii+1,2);
    colorBlue(labeledPicture==ii) = colors(ii+1,3);
end
colorImage = cat(3,colorRed,colorGreen,colorBlue);


%% Plotting
if plotOn == true
    
    % NDVI mesh
    % figure(1)
    % mesh(NDVI)
    
    % NDVI image
    figure(2)
    imshow(CutNDVI)
    title(['See all plants, NDVI= ',num2str(cutoffNDVI)])
    
    % Labeled & colored image
    figure(3)
    imshow(colorImage)
    title('Labeled regions colored')
    L = line(ones(length(classLabel)),ones(length(classLabel)), 'LineWidth',2);
    set(L,{'color'},mat2cell(colors./255,ones(1,length(classLabel)),3));
    legend(classNames,'Location','southeast');
    
end

%% Function testing

% isInsidePolygon([361,667],edgesCorn)

%% Functions

function x = isInsidePolygon(point, edges)
if size(edges,2)~= 2 || size(edges,1)< 3
    x=false;
    return
end
edges(end+1,:)=edges(1,:);
for i=2:length(edges)
    if isLeftOfLine(point,edges(i-1:i,:)) ~= true
        x=false;
        return
    end
end
x=true;
end

function x = isLeftOfLine(point, linePoints)
    grad = diff(linePoints);
    D = -grad(2) * ( point(1) - linePoints(1,1) ) + grad(1) * ( point(2) - linePoints(1,2) );
    if(D > 0)
        x=false;
        return
    end
    x=true;
end

