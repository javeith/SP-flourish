%% Loading images and computing NDVI 
% (Band8 - Band1) / (Band8 + Band1) = (803-700) / (803+700)
 cutoffNDVI = .51;

pathBand1 = '/Volumes/mac_jannic_2017/thanujan/Datasets/Ximea_Tamron/20170622/Orthomosaics/Band1.png';
pathBand8 = '/Volumes/mac_jannic_2017/thanujan/Datasets/Ximea_Tamron/20170622/Orthomosaics/Band8.png';

orthomosaicBand1 = im2double(rgb2gray(imread(pathBand1)));
orthomosaicBand8 = im2double(rgb2gray(imread(pathBand8)));

NDVI = (orthomosaicBand8 - orthomosaicBand1) ./ (orthomosaicBand1 + orthomosaicBand8);

figure(1)
mesh(NDVI)

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

%% Create labeling polygons via ordered edgepoints, counter clockwise

% Points [X,Y] are corners of a convex polygon, corner points must be
% ordered counter clockwise. Needs a minimum of 3 points.

% Ximea_Tamron/20170622
edgesCorn = [47,603;1181,331;1181,175;32,443];
edgesSugarbeet = [120,1009;987,805;987,409;21,641];
edgesWinterwheat = [441,1075;991,951;993,839;413,989];
edgesRoad = [];
edgesSoil = [];
edgesBuckwheat = [];
edgesGrass = [];
edgesSoybean = [];

%% Show threshholed input
showPicture=NDVI;
showPicture(NDVI>= cutoffNDVI) = 255;
showPicture(NDVI< cutoffNDVI) = 0;
figure(2)
imshow(showPicture)
title(['See all plants, NDVI= ',num2str(cutoffNDVI)])

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

figure(3)
imshow(labeledPicture)
title('Labeled Regions Blacked out')

save('/Users/jveith/Documents/2018 Semester Project/Python/trainingLabels.mat','labeledPicture')
%imwrite(labeledPicture,'/Users/jveith/Documents/2018 Semester Project/Python/trainingLabels.png')

%%% IN PROGRESS
% %% Create labeled color image
% colors= [cat(3,0,0,0);cat(3,0,0,255);cat(3,255,0,0);cat(3,0,255,0); ...
%     cat(3,255,128,0);cat(3,110,25,0);cat(3,125,0,255);cat(3,255,255,0);...
%     cat(3,0,137,255)];
% 
% colorImage = zeros(size(labeledPicture,1),size(labeledPicture,2),3);
% for ii= 1:8
%     colorImage(labeledPicture==ii,:) = colors(ii+1,:);
% end
% 
% figure(3)
% imshow(colorImage)
% title('Labeled regions colored')

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

