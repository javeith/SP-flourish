basePath = '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/Leonhard/';

resultList = dir([basePath,'*.mat']);
for i=1:size(resultList,1)
    FCNdisplayLabels([basePath,resultList(i).name],resultList(i).name);
    saveas(1,[basePath,'_',strtok(resultList(i).name,'.'),'.png'])
    close 1
end
