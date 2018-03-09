
sourcePath= '/Volumes/mac_jannic_2017/thanujan/Datasets/xFcnClassifier/testData/Ximea_Tamron/20170613/';

% load(['/Volumes/mac_jannic_2017/thanujan/Datasets/xClassifier/x41bands/PatternNet/net.mat']);

% testData =  im2double(imread([sourcePath,'Band1.png']));
% 
% for i=2:41
%     inPath = [sourcePath,'Band',num2str(i),'.png'];
%     temp = im2double(imread(inPath));
%     testData = cat(3,testData,temp);
% end
size1 = size(testData,1);
size2 = size(testData,2);

testData = reshape(testData,[size1*size2,41]);

result = net(testData);




% %% Plotting
% 
% % Names & colors
% classNames = {'Background','Corn','Sugarbeet','Winterwheat','Road','Soil',...
%     'Buckwheat','Grass','Soybean'};
% classNumbers = {0, 1, 2, 3, 4, 5, 6, 7, 8};
% classLabel = containers.Map(classNames,classNumbers);
% colors= [0,0,0;0,0,255;255,0,0;0,255,0;255,128,0;110,25,0;125,0,255;255,255,0;0,137,255];
% 
% % Call functions
% colorImage = fillColors(zeros(size(result),'uint8'), testLabelPredict, colors);
% plotColorImage( colorImage, classNames, classLabel, colors, imgTitle);
% 
% % Functions
% function plotColorImage(colorImage,classNames,classLabel,colors,imgTitle)
% figure()
% imshow(colorImage)
% title(imgTitle)
% L = line(ones(length(classLabel)),ones(length(classLabel)), 'LineWidth',2);
% set(L,{'color'},mat2cell(colors./255,ones(1,length(classLabel)),3));
% legend(classNames,'Location','southeast');
% end
% 
% function colorImage = fillColors(basePicture,labeledPicture,colors)
% colorRed = basePicture;
% colorGreen= basePicture;
% colorBlue = basePicture;
% for ii= 1:8
%     colorRed(labeledPicture==ii) = colors(ii+1,1);
%     colorGreen(labeledPicture==ii) = colors(ii+1,2);
%     colorBlue(labeledPicture==ii) = colors(ii+1,3);
% end
% colorImage = cat(3,colorRed,colorGreen,colorBlue);
% end
