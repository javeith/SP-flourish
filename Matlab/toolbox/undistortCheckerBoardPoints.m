function [ udCP ] = undistortCheckerBoardPoints( checkerboardPoints, cameraMatrix, radDist, tangDist )

% Convert grid values to normalised values
x = (checkerboardPoints(:,1)-cameraMatrix(1,3))/cameraMatrix(1,1);
y = (checkerboardPoints(:,2)-cameraMatrix(2,3))/cameraMatrix(2,2);

% Add distortion
r = x.^2 + y.^2;

% Testing with radtan
x_hd = (1 + radDist(1).*r + radDist(2).*r.^2 + radDist(3).*r.^3).*x + 2*tangDist(1).*x.*y + tangDist(2).*(r+2.*x.^2);
y_hd = (1 + radDist(1).*r + radDist(2).*r.^2 + radDist(3).*r.^3).*y + 2*tangDist(2).*x.*y + tangDist(1).*(r+2.*y.^2);

% Get pixel coordinates
xd = x_hd*cameraMatrix(1,1) + cameraMatrix(1,3);
yd = y_hd*cameraMatrix(2,2) + cameraMatrix(2,3);

udCP = [xd,yd];

end