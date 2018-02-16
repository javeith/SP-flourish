basePath = '/Volumes/mac_jannic_2017/thanujan/Datasets/';
outBase = '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testData/';

imagePath1NIR = [basePath , 'FIP/20170802/testSet/Orthomosaics/'];
imagePath1VIS = [basePath , 'FIP/20170802/testSet/VIS_Orthomosaics/'];
outPath1 = [outBase, 'FIP/20170802/'];

imagePath2NIR = [basePath , 'FIP/20170531/testSet/Orthomosaics/'];
imagePath2VIS = [basePath , 'FIP/20170531/testSet/VIS_Orthomosaics/'];
outPath2 = [outBase, 'FIP/20170531/'];

imagePath3NIR = [basePath , 'FIP/20170622/testSet/Orthomosaics/'];
imagePath3VIS = [basePath , 'FIP/20170622/testSet/VIS_Orthomosaics/'];
outPath3 = [outBase, 'FIP/20170622/'];

imagePath4NIR = [basePath , 'Ximea_Tamron/20170613/Orthomosaics/'];
imagePath4VIS = [basePath , 'Ximea_Tamron/20170613/VIS_Orthomosaics/'];
outPath4 = [outBase, 'Ximea_Tamron/20170613/'];

%FCNcreateTestImages(imagePath1NIR,imagePath1VIS,outPath1,...
%    true, [1600,1600]);
FCNcreateTestImages(imagePath2NIR,imagePath2VIS,outPath2,...
    true, [1600,1600]);
FCNcreateTestImages(imagePath3NIR,imagePath3VIS,outPath3,...
    true, [1600,1600]);
FCNcreateTestImages(imagePath4NIR,imagePath4VIS,outPath4,...
    true, [1600,1600]);