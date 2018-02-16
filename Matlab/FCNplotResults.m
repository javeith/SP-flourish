basePath = '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/';

labelPath1 = [ basePath, 'FIP_20170531.mat'];
labelPath2 = [ basePath, 'FIP_20170622.mat'];
labelPath3 = [ basePath, 'FIP_20170802.mat'];
labelPath4 = [ basePath, 'XimeaT_20170613.mat'];

FCNdisplayLabels(labelPath1,'FIP_20170531');
FCNdisplayLabels(labelPath2,'FIP_20170622');
FCNdisplayLabels(labelPath3,'FIP_20170802');
FCNdisplayLabels(labelPath4,'XimeaT_20170613');

saveas(1,'/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/FIP_20170531.png')
saveas(2,'/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/FIP_20170622.png')
saveas(3,'/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/FIP_20170802.png')
saveas(4,'/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/XimeaT_20170613.png')