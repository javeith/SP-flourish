% IN: Labels & corredponding data
% OUT: Characteristic spectral lines for the crops
% Counter: 41x8

global classNames sortedFrequencies index colors

classNames = {'Corn','Sugarbeet','Winterwheat','Road','Soil',...
    'Buckwheat','Grass','Soybean'};

frequencies = [615 623 608 791 686 816 828 803 791 700 765 778 752 739 ...
     714 653 662 645 636 678 867 864 857 845 670 465 546 586 630 474 534 578 ...
     624 485 522 562 608 496 510 548 600];

colors= [0,0,255;255,0,0;0,255,0;255,128,0;110,25,0;125,0,255;255,255,0;0,137,255]/255;

[sortedFrequencies,index] = sort(frequencies);
 
basePath = '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/';

dataPaths = {'trainData/FIP/20170531/',...
            'trainData/FIP/20170622/',...
            'trainData/Ximea_Tamron/20170510/',...
            'trainData/Ximea_Tamron/20170622/',...
            'testData/Ximea_Tamron/20170613/',...
            'testData/FIP/20170802/'};
        
labelPaths = {'trainLabels/FIP_20170531.mat',...
            'trainLabels/FIP_20170622.mat',...
            'trainLabels/Ximea_Tamron_20170510.mat',...
            'trainLabels/Ximea_Tamron_20170622.mat',...
            'testLabelTruth/Ximea_Tamron_20170613.mat',...
            'testLabelTruth/FIP_20170802.mat'};

allMeansForAllImages=[];

for i=1:6
    allMeansForAllImages = cat(3,allMeansForAllImages,computeMeansForOneImage([basePath,labelPaths{i}],[basePath,dataPaths{i}]));
end

oneMeanForAllImages = mean(allMeansForAllImages, 3,'omitnan');

%% Plotting

for i=1:6
    plotLines(allMeansForAllImages(:,:,i));
end
plotLines(oneMeanForAllImages);

%% Functions
function plotLines(matrix)

global classNames sortedFrequencies index colors

sortedMeans = matrix(index,:);
figure
for i=1:8
    plot(sortedFrequencies,sortedMeans(:,i),'-x','Color',colors(i,:),'LineWidth',2)
    hold on
end
legend(classNames,'Location','southeast');

end


function outMeans = computeMeansForOneImage(labelPath,dataPath)
label = loadLabel(labelPath);
for ii=1:41
    temp = imread([dataPath,'Band',num2str(ii),'.png']);
    for jj=1:8
        outMeans(ii,jj) = mean(temp(label==jj));
    end
end
end

function label = loadLabel(Path)
matArray = load(Path);
matArrayName = fields(matArray);
label = matArray.(matArrayName{1});
end
