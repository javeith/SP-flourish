function [ imagePoints ] = normalizedToPixel( normalizedPoints, imageWidth, imageHeight )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

imagePoints = (normalizedPoints+1) .* [(imageWidth-1), (imageHeight-1)] ./2 +1;

end

