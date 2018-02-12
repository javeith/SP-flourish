%
% Jannic Veith
% Autonomous Systems Lab
% ETH Zürich
%
%% Method description
% 
% This script labels all points with NDVI >= cutoffNDVI with labels 0 to 8.
% For Soil, Road and Background the condition is NDVI < cutoffNDVI.
% See Labeling for more info.
% 
% First select the desired label in the dropdown menu, then choose the
% limiting corners in COUNTER-CLOCKWISE direction. Confirm selection of
% points with ENTER,remove last selected point with BACKSPACE.
% 
% Input orthomosaic paths are pathBand1 and pathBand8.
% The labeled image will be saved to the path savePath as a .mat file.
%

%% Set options
clc
clear all;
close all;

inPathNIR = '/Volumes/mac_jannic_2017/thanujan/Datasets/Ximea_Tamron/20170510/Orthomosaics/';
inPathVIS = '/Volumes/mac_jannic_2017/thanujan/Datasets/Ximea_Tamron/20170510/VIS_Orthomosaics/';

% Paths for orthomosaics for Band1 (700nm) and Band8 (803nm)
pathBand1 = '/Volumes/mac_jannic_2017/thanujan/Datasets/Ximea_Tamron/20170510/Orthomosaics/Band1.png';
pathBand8 = '/Volumes/mac_jannic_2017/thanujan/Datasets/Ximea_Tamron/20170510/Orthomosaics/Band8.png';

% Path to save the labeled data as .mat file
saveDataPath = '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/Data/';
saveLabelPath = '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/Labels/XimeaT20170510.mat';

% Cutoff NDVI for best separation of plants and soil. If you want to see
% the NDVI image for point selection, set cutoffON to false. The data
% labeling still uses the cutoffNDVI when cutoffON is false.
cutoffNDVI = .505;
cutoffON = false;

% adjustData=True: For all unlabeled pixels set image pixel intensities to 0
adjustData = false;


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

% Colors for plotting
colors= [0,0,0;0,0,255;255,0,0;0,255,0;255,128,0;110,25,0;125,0,255;255,255,0;0,137,255];


%% Load and initialize images

% Loading images and computing NDVI
orthomosaicBand1 = im2double(rgb2gray(imread(pathBand1)));
orthomosaicBand8 = im2double(rgb2gray(imread(pathBand8)));

% (Band8 - Band1) / (Band8 + Band1) = (803-700) / (803+700)
NDVI = (orthomosaicBand8 - orthomosaicBand1) ./ (orthomosaicBand1 + orthomosaicBand8);

% Threshholding for plotting
CutNDVI=im2uint8(NDVI);
if cutoffON
    CutNDVI(NDVI>= cutoffNDVI) = 255;
    CutNDVI(NDVI< cutoffNDVI) = 0;
end

labeledPicture = zeros(size(NDVI,1),size(NDVI,2),1);


%% Keep looping for every label the user wants to set.

while 1
    colorPicture = fillColors(CutNDVI,labeledPicture,colors);
    plotColorImage(1,colorPicture,classNames,classLabel,colors)
    label = choosedialog(classNames);
    if strcmp(label, 'done' )
        break;
    end
    figure(2)
    [edges(:,1),edges(:,2),~] = impixel(colorPicture);
    labeledPicture = labelPicture(labeledPicture,edges,label,classLabel,cutoffNDVI,NDVI);
    clear edges
end
if ishandle(2)
    close(2);
end

save(saveLabelPath,'labeledPicture');

% Modifies the images according to labeledPicture
if adjustData
    adjustImageData(inPathNIR,inPathVIS,saveDataPath,labeledPicture);
end


%% Function testing

% isInsidePolygon([361,667],edgesCorn)


%% Functions

function adjustImageData(inPathNIR,inPathVIS,saveDataPath,labeledPicture)

for i=1:25
    temp = imread([inPathNIR,'Band',num2str(i),'.png']);
    temp( labeledPicture==0 ) =0;
    imwrite(temp,[saveDataPath,'Band',num2str(i),'.png']);
    clear temp
    if i<17
        temp = imread([inPathVIS,'Band',num2str(i),'.png']);
        temp( labeledPicture==0 ) =0;
        imwrite(temp,[saveDataPath,'Band',num2str(i+25),'.png']);
        clear temp
    end
end

end

function plotColorImage(figureNumber,colorImage,classNames,classLabel,colors)

figure(figureNumber)
imshow(colorImage)
title('Labeled regions colored')
L = line(ones(length(classLabel)),ones(length(classLabel)), 'LineWidth',2);
set(L,{'color'},mat2cell(colors./255,ones(1,length(classLabel)),3));
legend(classNames,'Location','southeast');
end
function colorImage = fillColors(basePicture,labeledPicture,colors)

colorRed = basePicture;
colorGreen= basePicture;
colorBlue = basePicture;
for ii= 1:8
    colorRed(labeledPicture==ii) = colors(ii+1,1);
    colorGreen(labeledPicture==ii) = colors(ii+1,2);
    colorBlue(labeledPicture==ii) = colors(ii+1,3);
end
colorImage = cat(3,colorRed,colorGreen,colorBlue);

end

function outPicture =labelPicture(inPicture,edges,label,classLabel,cutoffNDVI,NDVI)
outPicture = inPicture;
for ii=1:size(inPicture,1)
    for jj=1:size(inPicture,2)
        if strcmp(label,'Soil') || strcmp(label,'Background') || strcmp(label,'Road')
            if NDVI(ii,jj) < cutoffNDVI
                if isInsidePolygon([jj,ii],edges) == true
                    outPicture(ii,jj)=classLabel(label);
                end
            end
        else
            if NDVI(ii,jj) >= cutoffNDVI
                if isInsidePolygon([jj,ii],edges) == true
                    outPicture(ii,jj)=classLabel(label);
                end
            end
        end
    end
end

end
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

function choice = choosedialog(choosableParameters)

d = dialog('Position',[300 300 250 150],'Name','Select label');

txt = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[5 80 190 40],...
    'String','Select the label you wish to add:');

popup = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[20 60 200 25],...
    'String',choosableParameters,...
    'Callback',@popup_callback);

btn = uicontrol('Parent',d,...
    'Position',[20 20 60 25],...
    'String','Accept',...
    'Callback','delete(gcf)');

btn = uicontrol('Parent',d,...
    'Position',[140 20 80 25],...
    'String','Stop and save',...
    'Callback',@button_callback);

choice = 'done';

% Wait for d to close before running to completion
uiwait(d);

    function popup_callback(popup,event)
        idx = popup.Value;
        popup_items = popup.String;
        choice = char(popup_items(idx,:));
    end

    function button_callback(src,event)
        choice = 'done';
        delete(gcf);
    end
end

