%% Estimate DJI <--> Ximea transformation 
clc
clear
% close all

%% Inputs:
% - Camera poses DJI and ximea:
load /home/thanu/Documents/CWG-CALTag2/shortBag/DJIPoses.mat
load /home/thanu/Documents/CWG-CALTag2/shortBag/ximeaPoses.mat

load /home/thanu/Documents/CWG-CALTag2/shortBag/DJIOrientLoc.mat
load /home/thanu/Documents/CWG-CALTag2/shortBag/ximeaOrientLoc.mat


% - Corresponding gimbal angles for each ximea image:
load /home/thanu/Documents/CWG-CALTag2/shortBag/angleIDtoEachXimea.mat
load /home/thanu/Documents/CWG-CALTag2/shortBag/gimbalPYRT.mat

% Matches:
load /home/thanu/Documents/CWG-CALTag2/shortBag/matchDJIXimea.mat
% 15-19

%% Calculate gimbal angles rotation matrices
T_gimbalAngles = zeros(4,4,5);
T_gimbalAngles(4,4,:) = 1;

for i = 2:6
    yaw = gimbalPYRT(matchDJIXimea(i,4),2);
    pitch = gimbalPYRT(matchDJIXimea(i,4),1) - gimbalPYRT(1,1);
    roll = gimbalPYRT(matchDJIXimea(i,4),3) - gimbalPYRT(1,3);
    
    T_gimbalAngles(1:3,1:3,i-1) = DJIPoses(1:3,1:3,matchDJIXimea(i,1)) * eul2rotm([degtorad(yaw), degtorad(pitch), degtorad(roll)]);
%     T_gimbalAngles(1:3,1:3,i-1) = eye(3);
end

clear yaw pitch roll gimbalPYRT correspondingGimbalAngles

%% Extract needed transformations
% Transformation caltag frame to actuated camera frame
T_FA = zeros(4,4,5);

for i = 2:6
   T_AF(:,:,i-1) = DJIOrientLoc(:,:,matchDJIXimea(i,1)); 
   T_FA(:,:,i-1) = [T_AF(1:3,1:3)', (-T_AF(1:3,1:3))'*T_AF(1:3,4);0,0,0,1];
end

% Transformation caltag frame to static camera frame
T_SF = zeros(4,4,5);

for i = 2:6
   T_SF(:,:,i-1) = ximeaOrientLoc(:,:,matchDJIXimea(i,3));
end

T_SA = [(T_SF(1:3,1:3,1)*T_AF(1:3,1:3,1))', (T_SF(1:3,4,1)-T_AF(1:3,4,1)); 0,0,0,1];
T_SA(1:3,4) = T_SA(1:3,4)./1000;

clear T_AF ximeaPoses DJIPoses

%% Optimization to find constant part of transformation
T_constant_init = eye(4);
T_constant = eye(4);

options = optimoptions('fminunc', 'MaxFunctionEvaluations', 2500, 'Algorithm',...
    'quasi-newton','OptimalityTolerance', 1.0000e-20,'StepTolerance', 1.0000e-20);

err_func_R = @(tM) optProblem_R(T_FA(:,:,1),T_SF(:,:,1),eye(4),tM);
[T_constant(1:3,1:3),err_R] = fminunc(err_func_R, T_constant_init(1:3,1:3),options);

R_constant = T_constant(1:3,1:3);
err_func_t = @(tM) optProblem_t(T_FA(:,:,1),T_SF(:,:,1),eye(4),R_constant,tM);
[T_constant(1:3,4),err_t] = fminunc(err_func_t, T_constant_init(1:3,4),options);

rotm2eul(T_constant(1:3,1:3))*(180/pi)

% T_FA(:,:,2) * T_gimbalAngles(:,:,2) * T_constant * T_SF(:,:,2)

T_constant = [T_constant(1:3,1:3)', (-T_constant(1:3,1:3))'*T_constant(1:3,4);0,0,0,1];
T_constant(1:3,4) = T_constant(1:3,4)./1000;

