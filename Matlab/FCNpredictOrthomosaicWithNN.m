close all
clear all
clc

sourcePath= '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testData/FIP/20170802/';
savePath= '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/MatlabNNPredict/FIP_20170802';

load(['/Volumes/mac_jannic_2017/thanujan/Datasets/xClassifier/x41bands/PatternNet/net.mat']);

testData =  double(imread([sourcePath,'Band1.png']));

for i=2:41
    inPath = [sourcePath,'Band',num2str(i),'.png'];
    temp = double(imread(inPath));
    testData = cat(3,testData,temp);
end

% temp1 =  double(imread([sourcePath,'Orthomosaics/','Band1.png']));
% testData = temp1(:,:,1);
% 
% for i=2:41
%     if i<=25
%         inPath = [sourcePath,'Orthomosaics/','Band',num2str(i),'.png'];
%         temp = (double(imread(inPath)));
%         testData = cat(3,testData,temp(:,:,1));
%     else
%         inPath = [sourcePath,'VIS_Orthomosaics/','Band',num2str(i-25),'.png'];
%         temp = (double(imread(inPath)));
%         testData = cat(3,testData,temp(:,:,1));
%     end
% end

size1 = size(testData,1);
size2 = size(testData,2);

testData = reshape(testData,[size1*size2,41]);

result = net(transpose(testData));
[~,binaryResult] = max(result);
Prediction = reshape(transpose(binaryResult),[size1,size2]);
Prediction = relabel(Prediction,sourcePath);
plotPredictions(Prediction);

save([savePath,'.mat'],'Prediction');
saveas(1,[savePath,'.png']);

%% Functions
function outIMG = relabel(resultIMG,sourcePath)
% Relabel NN output to my (usual) indexing. (see classnames below)
outIMG = zeros(size(resultIMG));

backgroundIndex =  double(imread([sourcePath,'Band1.png']));
outIMG(resultIMG==1) = 5;
outIMG(resultIMG==2) = 4;
outIMG(resultIMG==3) = 6;
outIMG(resultIMG==4) = 1;
outIMG(resultIMG==5) = 7;
outIMG(resultIMG==6) = 8;
outIMG(resultIMG==7) = 2;
outIMG(resultIMG==8) = 3;
outIMG(backgroundIndex==0)=0;

end

function plotPredictions(plotImg)
% Prediction output labeling from 1-8:
% [soil, road, buckWheat, corn, grass, soyBean, sugarBeet, winterWheat]

% Names & colors
classNames = {'Background','Corn','Sugarbeet','Winterwheat','Road','Soil',...
    'Buckwheat','Grass','Soybean'};
classNumbers = {0, 1, 2, 3, 4, 5, 6, 7, 8};
classLabel = containers.Map(classNames,classNumbers);
colors= [0,0,0;0,0,255;255,0,0;0,255,0;255,128,0;110,25,0;125,0,255;255,255,0;0,137,255];

% Call functions
colorImage = fillColors(zeros(size(plotImg),'uint8'), plotImg, colors);
plotColorImage( colorImage, classNames, classLabel, colors, 'NN Prediction');

% Functions
function plotColorImage(colorImage,classNames,classLabel,colors,imgTitle)
figure()
imshow(colorImage)
title(imgTitle)
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

end
