close all

basePath = '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/Leonhard/';
%basePath = '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/';
%basePath = '/Users/jveith/Documents/2018 Semester Project/Python/LeonhardData/';


resultList = dir([basePath,'*.mat']);
for i=1:size(resultList,1)
    FCNdisplayLabels([basePath,resultList(i).name],resultList(i).name);
    saveas(1,[basePath,'_',strtok(resultList(i).name,'.'),'.png'])
    close 1
end
