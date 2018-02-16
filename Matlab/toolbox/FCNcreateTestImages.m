function []= createTestImagesFCN(inPathNIR,inPathVIS,outDataPath,addPadding, finalSize)

mkdir(outDataPath);
for i=1:25
    inPath = [inPathNIR,'Band',num2str(i),'.png'];
    outPath = [outDataPath,'Band',num2str(i),'.png'];
    createAndSaveOneTrainingImage(inPath, outPath, addPadding, finalSize);
    if i<17
        inPath = [inPathVIS,'Band',num2str(i),'.png'];
        outPath = [outDataPath,'Band',num2str(i+25),'.png'];
        createAndSaveOneTrainingImage(inPath, outPath, addPadding, finalSize);
    end
end



function createAndSaveOneTrainingImage(inPath, outPath, ...
    addPadding, finalSize)
temp = im2double(rgb2gray(imread(inPath)));
if addPadding
    temp = addPaddingToImage(temp,finalSize);
end
imwrite(temp(:,:,1),outPath);
end

function paddedImage = addPaddingToImage(image,finalSize)
paddedImage = zeros(finalSize(2),finalSize(1));

xLowerPadding = ceil( (finalSize(1) - size(image,2))/2 );
yLowerPadding = ceil( (finalSize(2) - size(image,1))/2 );

yIndex1 = yLowerPadding+1;
yIndex2 = yLowerPadding+size(image,1);
xIndex1 = xLowerPadding+1;
xIndex2 = xLowerPadding+size(image,2);

paddedImage(yIndex1:yIndex2, xIndex1:xIndex2) = image;
end

end