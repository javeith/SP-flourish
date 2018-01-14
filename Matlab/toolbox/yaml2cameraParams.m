function [ params ] = yaml2cameraParams( yaml )
%yaml2cameraParams Convert ROS yaml camera calibration to MATLAB camera
%parameters

intrinsics = ReadYaml(yaml);
cM(1,:) = cell2mat(intrinsics.camera_matrix.data(1:3));
cM(2,:) = cell2mat(intrinsics.camera_matrix.data(4:6));
cM(3,:) = cell2mat(intrinsics.camera_matrix.data(7:9));

params = toStruct(cameraParameters);
params.IntrinsicMatrix = cM';

params.RadialDistortion = cell2mat(intrinsics.distortion_coefficients.data(1:3));
params.TangentialDistortion = cell2mat(intrinsics.distortion_coefficients.data(4:end));

params.NumRadialDistortionCoefficients = 3;
params.EstimateTangentialDistortion = true;

%   Detailed explanation goes here


end

