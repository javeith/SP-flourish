basePath = '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/';

labelPath1 = [ basePath, 'FIP_20170531.mat'];
labelPath2 = [ basePath, 'FIP_20170622.mat'];
labelPath3 = [ basePath, 'FIP_20170802.mat'];
labelPath4 = [ basePath, 'Ximea_Tamron_20170613.mat'];
labelPath5 = [ basePath, 'Ximea_Tamron_20170510.mat'];
labelPath6 = [ basePath, 'Ximea_Tamron_20170622.mat'];

FCNdisplayLabels(labelPath1,'FIP_20170531');
FCNdisplayLabels(labelPath2,'FIP_20170622');
FCNdisplayLabels(labelPath3,'FIP_20170802');
FCNdisplayLabels(labelPath4,'Ximea_Tamron_20170613');
FCNdisplayLabels(labelPath5, 'Ximea_Tamron_20170510.mat');
FCNdisplayLabels(labelPath6, 'Ximea_Tamron_20170622.mat');

saveas(1,'/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/pFIP_20170531.png')
saveas(2,'/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/pFIP_20170622.png')
saveas(3,'/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/pFIP_20170802.png')
saveas(4,'/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/pXimea_Tamron_20170613.png')
saveas(5,'/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/pXimea_Tamron_20170510.png')
saveas(6,'/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testLabelPredict/pXimea_Tamron_20170622.png')

