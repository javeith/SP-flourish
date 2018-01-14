function [ unImg ] = undistort( img, cameraMatrix, distortion, model )
% model = [radtan4, radtan5, fisheye, equidistant]

% Generate grid of ideal values in the undistorted image
[rows,cols,channel] = size(img);
[xu,yu] = meshgrid(1:cols, 1:rows);

% Convert grid values to normalised values
x = (xu-cameraMatrix(1,3))/cameraMatrix(1,1);
y = (yu-cameraMatrix(2,3))/cameraMatrix(2,2);

% Add distortion
r = x.^2 + y.^2;

% Testing with radtan
if strcmp(model, 'radtan5')
    x_hd = (1 + distortion(1).*r + distortion(2).*r.^2 + distortion(3).*r.^3).*x + 2*distortion(4).*x.*y + distortion(5).*(r+2.*x.^2);
    y_hd = (1 + distortion(1).*r + distortion(2).*r.^2 + distortion(3).*r.^3).*y + 2*distortion(5).*x.*y + distortion(4).*(r+2.*y.^2);
    
elseif strcmp(model, 'radtan4')
    x_hd = (1 + distortion(1).*r + distortion(2).*r.^2).*x + 2*distortion(3).*x.*y + distortion(4).*(r+2.*x.^2);
    y_hd = (1 + distortion(1).*r + distortion(2).*r.^2).*y + 2*distortion(4).*x.*y + distortion(3).*(r+2.*y.^2);
    
elseif strcmp(model, 'fisheye')
    theta = 2/pi * atan(sqrt(r));
    theta = (distortion(2).*theta + distortion(3).*theta.^2 + distortion(4).*theta.^3 + distortion(5).*theta.^4);
    x_hd = theta.*x ./ sqrt(x.^2 + y.^2);
    y_hd = theta.*y ./ sqrt(x.^2 + y.^2);
    
elseif strcmp(model, 'equidistant')
    theta = atan(sqrt(r));
    theta = (theta + distortion(1)*theta.^3 + distortion(2)*theta.^5 ...
        + distortion(3)*theta.^7 + distortion(4)*theta.^9);
    omega = atan2(y,x);
    x_hd = theta.*cos(omega);
    y_hd = theta.*sin(omega);
end

% Get pixel coordinates
xd = x_hd*cameraMatrix(1,1) + cameraMatrix(1,3);
yd = y_hd*cameraMatrix(2,2) + cameraMatrix(2,3);

% Interpolate values from distorted image to their ideal locations
unImg = zeros(size(img));
for n = 1:channel
    unImg(:,:,n) = interp2(xu,yu,double(img(:,:,n)),xd,yd);
end

% Cast back to uint8
if isa(img, 'uint8')
    unImg = uint8(unImg);
    
elseif isa(img, 'uint16')
    unImg = uint16(unImg);
end

end