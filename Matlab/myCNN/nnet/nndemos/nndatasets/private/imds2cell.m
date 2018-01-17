function [X, T] = imds2cell(imds)
% imds2cell   Convert an image datastore into a cell array of images
%
%   X       - Input data as a cell array of images of size 1-by-N.
%   T       - Targets as a K-by-N matrix, where K is the number
%             of classes and N is the number of images.

% Copyright 2016 The MathWorks, Inc.

X = imds.readall();
X = cellfun( @im2double, X, 'UniformOutput', false ); % Convert X to double
X = X'; % Make X 1-by-N

T = iDummify( imds.Labels );
end

function dummifiedOut = iDummify( categoricalIn )
numObservations = numel(categoricalIn);
numCategories = numel(categories(categoricalIn));
dummifiedSize = [numCategories, numObservations];
dummifiedOut = zeros(dummifiedSize);
categoricalIn = iMakeHorizontal( categoricalIn );
idx = sub2ind(dummifiedSize, int32(categoricalIn), 1:numObservations);
dummifiedOut(idx) = 1;

% 0 has to become 10 (0000000001)
dummifiedOut = circshift(dummifiedOut, -1, 1);
end

function vec = iMakeHorizontal( vec )
vec = reshape( vec, 1, numel( vec ) );
end