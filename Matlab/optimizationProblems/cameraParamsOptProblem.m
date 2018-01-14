function [ err ] = cameraParamsOptProblem(x, R_10, t_01, worldPoints, imagePoints)

% cameraParams = cameraParameters('IntrinsicMatrix', [x(1),0,0;0,x(1),0;x(2),x(3),1]);
cM = [x(1),0,x(2);0,x(1),x(3);0,0,1];
R_21 = [x(4:6);x(7:9);x(10:12)];
t_12 = x(13:15)';

iP = zeros(size(imagePoints));

for i = 1:size(imagePoints,1)
    temp = cM * (R_21 * ([R_10, -R_10 * t_01] * [worldPoints(i,:)';1]) + t_12);
    iP(i,:) = [temp(1)/temp(3),temp(2)/temp(3)];
end
% iP = worldToImage(cameraParams,(R_21*R_10),-(R_10)*(t_01) + t_12,worldPoints);
tmp = iP - imagePoints;
err = sum(tmp(:,1).^2) + sum(tmp(:,2).^2);

end

