%% Description
%
% This function evaluates class predictions on 2D orthomosaics related to
% the Flourish framework.
%
% IN:   Folder paths of predictions and groundtruth.
%       For every .mat in groundtruth, one with the same name must be
%       present in the prediction folder. Only these .mar are used.
%
% OUT:  Plots of Confusion, Precision-Recall and groundtruth/prediction.
%
%
close all
clear all
clc
%% Sources

truthPath= '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelTruth/';
predictPath= '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/Leonhard/';%Archive/CNN64/';
%predictPath= '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/MatlabNNPredict/';

%% Load files and plot

truthFileNameList = dir([truthPath,'*.mat']);
predictFileNameList = dir([predictPath,'*.mat']);

labelContainer=[];
truthSet = [];
predictSet = [];

for truthFile = {truthFileNameList.name}
    tempPair = loadLabelPair([truthPath,truthFile{1}],[predictPath,truthFile{1}] );
    labelContainer = cat(4, labelContainer, tempPair);
    
    truthSet = cat(1, truthSet, reshape(tempPair(:,:,1), [size(tempPair,1)*size(tempPair,2),1]) );
    predictSet = cat(1, predictSet, reshape(tempPair(:,:,2), [size(tempPair,1)*size(tempPair,2),1]) );
end

% groundtruth/prediction
plotTruthPredict(labelContainer,{truthFileNameList.name});
clear labelContainer

% % % Precision-Recall
% truthSet = double(truthSet(truthSet~=0));
% predictSet = double(predictSet(truthSet~=0));
% plotPrecRecall(predictSet,truthSet);

truthSetNo0 = truthSet(truthSet~=0);
predictSetNo0 = predictSet(truthSet~=0);
% Confusion
% truthSet = Myind2vec(truthSet');
% predictSet = Myind2vec(predictSet');
% plotConfusion(truthSet,predictSet);

truthSet1 = Myind2vec1(truthSetNo0');
predictSet1 = Myind2vec1(predictSetNo0');
plotConfusion(truthSet1,predictSet1);

C = confusionmat(truthSet(truthSet~=0),predictSet(truthSet~=0));
B = C(2:end,2:end);
rec = diag(B)/sum(B,1);
prec = diag(B)./sum(B,2);

F1score = 2/(1/mean(rec)+1/mean(prec))


%% Functions
function plotTruthPredict(imageContainer,nameArray)

numberOfPairs = size(imageContainer,4);

% Names & colors
classNames = {'Background','Corn','Sugarbeet','Winterwheat','Road','Soil',...
    'Buckwheat','Grass','Soybean'};
classNumbers = {0, 1, 2, 3, 4, 5, 6, 7, 8};
classLabel = containers.Map(classNames,classNumbers);
colors= [0,0,0;0,0,255;255,0,0;0,255,0;255,128,0;110,25,0;125,0,255;255,255,0;0,137,255];

figure()
counter=0;

for i = 1:numberOfPairs
    
    subplot(numberOfPairs,2,i+counter)
    tempColor = fillColors(imageContainer(:,:,1,i),colors);
    imshow(tempColor)
    title(['Ground truth: ',nameArray{i}], 'Interpreter', 'none')
    L = line(ones(length(classLabel)),ones(length(classLabel)), 'LineWidth',2);
    set(L,{'color'},mat2cell(colors./255,ones(1,length(classLabel)),3));
    legend(classNames,'Location','southeast');
    
    counter=counter+1;
    
    subplot(numberOfPairs,2,i+counter)
    tempColor = fillColors(imageContainer(:,:,2,i),colors);
    imshow(tempColor)
    title(['Prediction: ',nameArray{i}], 'Interpreter', 'none')
    L = line(ones(length(classLabel)),ones(length(classLabel)), 'LineWidth',2);
    set(L,{'color'},mat2cell(colors./255,ones(1,length(classLabel)),3));
    legend(classNames,'Location','southeast');
    
end


    function colorImage = fillColors(labeledPicture,colors)
        colorRed = zeros(size(labeledPicture),'uint8');
        colorGreen= zeros(size(labeledPicture),'uint8');
        colorBlue = zeros(size(labeledPicture),'uint8');
        for ii= 1:8
            colorRed(labeledPicture==ii) = colors(ii+1,1);
            colorGreen(labeledPicture==ii) = colors(ii+1,2);
            colorBlue(labeledPicture==ii) = colors(ii+1,3);
        end
        colorImage = cat(3,colorRed,colorGreen,colorBlue);
    end
end

function labelPair = loadLabelPair(truthLabelPath,predictLabePath)

truthMat = load(truthLabelPath);
truthMatNames = fields(truthMat);
predictMat = load(predictLabePath);
predictMatNames = fields(predictMat);

truthImg = truthMat.(truthMatNames{1});
predictImg = predictMat.(predictMatNames{1});

labelPair = cat(3, truthImg, predictImg);
end


function plotConfusion(truth,prediction)
figure()
plotconfusion(truth,prediction)

end

function plotPrecRecall(p, t)

classNames = {'Background','Corn','Sugarbeet','Winterwheat','Road','Soil',...
    'Buckwheat','Grass','Soybean'};

for i = 1:size(t,1)
    [PREC, TPR, FPR, THRESH] = prec_rec(p(i,:),t(i,:),'holdFigure', 1, 'plotROC', 0, 'plotPR', 1, 'numThresh', 1000, 'plotBaseline', 0);
    f1Scores = 2*(PREC.*TPR)./(PREC+TPR);
    f1Score(i) = max(f1Scores);
end
mean(f1Score)
sqrt(var(f1Score))

% prec_rec(p,t);

grid on
title('Precision-Recall Curve')    
legend(classNames,'Location','southeast');

end

function out = Myind2vec(matrix)
out = zeros(9,size(matrix,2));
for i = 1:size(matrix,2)
    %if(matrix(i)~=0)
        out(matrix(i)+1, i) =1;
    %end
end
end

function out = Myind2vec1(matrix)
out = zeros(8,size(matrix,2));
for i = 1:size(matrix,2)
    if(matrix(i)~=0)
        out(matrix(i), i) =1;
    end
end
end