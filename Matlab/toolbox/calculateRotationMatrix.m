clc, clear 

%% Calculate relative rotation matrix for extrinsics.yaml

% Radian
rotX = -0.5418 * pi/180;
rotY = -0.1642 * pi/180;
rotZ = 0.1748 * pi/180;

% R1 = eul2rotm([rotZ,rotY,rotX]);

R = [cos(rotZ)*cos(rotY), -sin(rotZ)*cos(rotY), sin(rotY);...
    cos(rotZ)*sin(rotX)*sin(rotY)+sin(rotZ)*cos(rotX), cos(rotZ)*cos(rotX)-sin(rotZ)*sin(rotX)*sin(rotY), -sin(rotX)*cos(rotY);...
    sin(rotZ)*sin(rotX)-sin(rotY)*cos(rotZ)*cos(rotX), sin(rotZ)*cos(rotX)*sin(rotY)+cos(rotZ)*sin(rotX), cos(rotX)*cos(rotY)];