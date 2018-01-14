function [ image ] = readImageROS( bag, iImage)
%readImage Reads and returns a grayscale/color image from a rosbag
%   Detailed explanation goes here

%% Create selections for images and exposure times
% imgBag = select(bag, 'Topic', topic);

%% read image
imgMsg = readMessages(bag, iImage);
image = readImage(imgMsg{1});

end

