function displayFCNLabels(imagePath,imgTitle)

% Names & colors
classNames = {'Background','Corn','Sugarbeet','Winterwheat','Road','Soil',...
    'Buckwheat','Grass','Soybean'};
classNumbers = {0, 1, 2, 3, 4, 5, 6, 7, 8};
classLabel = containers.Map(classNames,classNumbers);
colors= [0,0,0;0,0,255;255,0,0;0,255,0;255,128,0;110,25,0;125,0,255;255,255,0;0,137,255];

% Call functions
matArray = load(imagePath);
matArrayName = fields(matArray);

colorImage = fillColors(zeros(size(matArray.(matArrayName{1})),'uint8'), matArray.(matArrayName{1}), colors);
plotColorImage( colorImage, classNames, classLabel, colors, imgTitle);

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