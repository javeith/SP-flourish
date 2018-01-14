function [ normalizedPoints ] = normalizedImagePoints( imagePoints, imageWidth, imageHeight)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

normalizedPoints = -1 + 2*(imagePoints - 1) ./ [(imageWidth-1), (imageHeight-1)];

end

