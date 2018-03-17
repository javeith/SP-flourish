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
% If createTraining==true, new images at location saveDataPath will be
% created where the intensity of every pixel with label 0 (background) will
% be set to 0.

%% Set options
% clc
% clear all;
% close all;

basePath = '/Volumes/mac_jannic_2017/thanujan/Datasets/';

inPathNIR = [ basePath, 'Ximea_Tamron/20170613/Orthomosaics/' ];
inPathVIS = [ basePath, 'Ximea_Tamron/20170613/VIS_Orthomosaics/' ];

% Path to save the labeled data as .mat file
saveLabelPath = [ basePath, 'xFcnClassifier/testLabelTruth/Ximea_Tamron_20170613' ];

% Cutoff NDVI for best separation of plants and soil. If you want to see
% the NDVI image for point selection, set cutoffON to false. The data
% labeling still uses the cutoffNDVI when cutoffON is false. Can have 1 or
% 2 values.
NDVIthresh = [.41, .48];
cutoffON = true;

% createTraining=True: Create image set where intensity = 0 if label = 0 at
% location saveDataPath
createTraining = false;
saveDataPath = [ basePath, 'xFcnClassifier/trainData/FIP/20170531/' ];

% Padding
addPadding = true;
finalSize = [1600,1600]; % [X,Y]
    

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
orthomosaicBand1 = im2double(rgb2gray(imread([inPathNIR,'Band1.png'])));
orthomosaicBand8 = im2double(rgb2gray(imread([inPathNIR,'Band8.png'])));

% (Band8 - Band1) / (Band8 + Band1) = (803-700) / (803+700)
NDVI = (orthomosaicBand8 - orthomosaicBand1) ./ (orthomosaicBand1 + orthomosaicBand8);

clear orthomosaicBand1 orthomosaicBand8

% Threshholding for plotting
switch length(NDVIthresh)
    case 1
        lowerThresh = NDVIthresh;
        upperThresh = NDVIthresh;
    case 2
        lowerThresh = NDVIthresh(1);
        upperThresh = NDVIthresh(2);
end

CutNDVI=im2uint8(NDVI);
if cutoffON
    CutNDVI(NDVI>= upperThresh) = 255;
    CutNDVI(NDVI< upperThresh) = 120;
    CutNDVI(NDVI< lowerThresh) = 0;
end

% Plot histogram
figure(1)
imhist(NDVI)
yl = ylim;

line([lowerThresh,lowerThresh],ylim,'Color','red','LineWidth',1)
line([upperThresh,upperThresh],ylim,'Color','red','LineWidth',1)

% figure(10)
% imshow(CutNDVI)

labeledPicture = zeros(size(NDVI,1),size(NDVI,2),1,'uint8');

%% Main: Keep looping for every label the user wants to set.

while 1
    colorPicture = fillColors(CutNDVI,labeledPicture,colors);
    plotColorImage(2,colorPicture,classNames,classLabel,colors)
    label = chooseGUI(classNames);
    if strcmp(label, 'done' )
        break;
    end
    figure(3)
    [edges(:,1),edges(:,2),~] = impixel(colorPicture);
    labeledPicture = labelPicture(labeledPicture,edges,label,classLabel,NDVIthresh,NDVI);
    clear edges
end
if ishandle(3)
    close(3);
end
if createTraining
    createTrainingData(inPathNIR,inPathVIS,saveDataPath,labeledPicture,addPadding,finalSize);
end
if addPadding
    labeledPicture = addPaddingToImage(labeledPicture,finalSize);
end

% Save labels as .mat
% imwrite(labeledPicture,[saveLabelPath,'.png'])
save([saveLabelPath,'.mat'],'labeledPicture');


%% Function testing

% Padding
% testImage = addPaddingToImage(NDVI,finalSize);
% figure(1)
% imshow(NDVI)
% figure(2)
% imshow(testImage)

% Plot labels in color
% a =  fillColors(zeros(1600,1600,'uint8'),labeledPicture,colors);
% plotColorImage(2,a,classNames,classLabel,colors);


%% Functions

function paddedImage = addPaddingToImage(image,finalSize)
paddedImage = zeros(finalSize(2),finalSize(1));

xLowerPadding = ceil( (finalSize(1) - size(image,2))/2 );
yLowerPadding = ceil( (finalSize(2) - size(image,1))/2 );

yIndex1 = yLowerPadding+1;
yIndex2 = yLowerPadding+size(image,1);
xIndex1 = xLowerPadding+1;
xIndex2 = xLowerPadding+size(image,2);

paddedImage(yIndex1:yIndex2, xIndex1:xIndex2) = image;
end

function createTrainingData(inPathNIR,inPathVIS,outDataPath,labeledPicture,...
    addPadding, finalSize)
mkdir(outDataPath);
for i=1:25
    inPath = [inPathNIR,'Band',num2str(i),'.png'];
    outPath = [outDataPath,'Band',num2str(i),'.png'];
    createAndSaveOneTrainingImage(labeledPicture, inPath, outPath, addPadding, finalSize);
    if i<17
        inPath = [inPathVIS,'Band',num2str(i),'.png'];
        outPath = [outDataPath,'Band',num2str(i+25),'.png'];
        createAndSaveOneTrainingImage(labeledPicture, inPath, outPath, addPadding, finalSize);
    end
end

end

function createAndSaveOneTrainingImage(labeledPicture, inPath, outPath, ...
    addPadding, finalSize)
temp = im2double(rgb2gray(imread(inPath)));
temp( labeledPicture==0 ) = 0;
if addPadding
    temp = addPaddingToImage(temp,finalSize);
end
imwrite(temp(:,:,1),outPath);
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

function outPicture =labelPicture(inPicture,edges,label,classLabel,NDVIthresh,NDVI)

switch length(NDVIthresh)
    case 1
        lowerThresh = NDVIthresh;
        upperThresh = NDVIthresh;
    case 2
        lowerThresh = NDVIthresh(1);
        upperThresh = NDVIthresh(2);
end

outPicture = inPicture;
for ii=1:size(inPicture,1)
    for jj=1:size(inPicture,2)
        if strcmp(label,'Soil') || strcmp(label,'Background') || strcmp(label,'Road')
            if NDVI(ii,jj) < lowerThresh
                if isInsidePolygon([jj,ii],edges) == true
                    outPicture(ii,jj)=classLabel(label);
                end
            end
        else
            if NDVI(ii,jj) >= upperThresh
                if isInsidePolygon([jj,ii],edges) == true
                    outPicture(ii,jj)=classLabel(label);
                end
            end
            if NDVI(ii,jj) < lowerThresh
                if isInsidePolygon([jj,ii],edges) == true
                    outPicture(ii,jj)=classLabel('Soil');
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

function choice = chooseGUI(choosableParameters)

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

