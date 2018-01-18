clear all

Channels = 5;
imagePath = 'testImages/Lenna.png';


I = imread(imagePath);
I_gray_small = imresize(rgb2gray(I),.1);
imageSize = size(I_gray_small);
imageContainer = I_gray_small;
for i=1:Channels-1
   imageContainer=cat(2,imageContainer,I_gray_small); 
end
imageContainer = reshape(imageContainer,imageSize(1),imageSize(2),Channels);

layers = [imageInputLayer([imageSize(1) imageSize(2) Channels]);
          convolution2dLayer(5,20);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          fullyConnectedLayer(10);
          softmaxLayer();
          classificationLayer()]