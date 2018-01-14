%% Subscribe to gps and image topic

clc, clear all
addpath('/home/thanu/catkin_ws/src/Onboard-SDK-ROS/matlab_gen/msggen');
% ADD MATLAB TO PATH !!!

% rosgenmsg('/home/thanu/catkin_ws/src/Onboard-SDK-ROS'); %% REMOVE CMakeLists.txt !!!
%rosinit

% sub = rossubscriber('/falcon/dji_ros/gps');
% % sub = rossubscriber('/falcon/color/color');
% % sub = rossubscriber('/falcon/infrared/infrared');
%
%
% i = 1;
% while true
%     msg = receive(sub,10);
%     Latitude1(i) = msg.Latitude;
%     Longitude1(i) = msg.Longitude;
%     Altitude1(i) = msg.Altitude;
%     %time = rostime('now')
%     %time.Nsec;
%     timeGPS(i) = msg.Header.Stamp.Sec + msg.Header.Stamp.Nsec / 10^9;
%     seqGPS(i) = msg.Header.Seq
%
%     i = i + 1;
% end

%rosshutdown

%% - Inputs:
bag = rosbag('/home/thanu/Documents/CWG-CALTag2/shortBag/falcon-2017-06-20-17-35-53.bag');
imgTopic = select(bag,'Topic', '/ximea_asl/image_raw');
gimbalTopic = select(bag,'Topic', '/falcon/dji_sdk/gimbal');

%% Extract data

% Number of messages
NoI = imgTopic.NumMessages;
NoG = gimbalTopic.NumMessages;

% Extract publish time
% imageTime = imgTopic.MessageList.Time;

imageTime = zeros(NoI,1);

for i = 1: NoI
    imgMsg = readMessages(imgTopic, i);
    imageTime(i) = imgMsg{1}.Header.Stamp.Sec + imgMsg{1}.Header.Stamp.Nsec / (10^9);
end

gimbalPYRT = zeros(NoG,4);

for j = 1:NoG
    gimbalMsg = readMessages(gimbalTopic, j);
    gimbalPYRT(j,1) = gimbalMsg{1}.Pitch;
    gimbalPYRT(j,2) = gimbalMsg{1}.Yaw;
    gimbalPYRT(j,3) = gimbalMsg{1}.Roll;
    gimbalPYRT(j,4) = gimbalMsg{1}.Header.Stamp.Sec + gimbalMsg{1}.Header.Stamp.Nsec / (10^9);
end

%% Find closest gimbal index to each image

correspondingGimbalAngles = zeros(NoI, 2);
for i = 1: NoI
    % value to find
    val = imageTime(i);
    
    tmp = abs(gimbalPYRT(:,4)-val);
    
    % index of closest value
    [value, idx] = min(tmp);
    
    % closest index
    correspondingGimbalAngles(i,1) = i;
    correspondingGimbalAngles(i,2) = idx;
    
end

%% Extract gimbal angles 
Pitch = zeros(size(correspondingGimbalAngles,1),1);
Yaw = zeros(size(correspondingGimbalAngles,1),1);
Roll = zeros(size(correspondingGimbalAngles,1),1);

for k = 1:size(correspondingGimbalAngles,1)
    gimbalMsg = readMessages(gimbalTopic, correspondingGimbalAngles(k,2));
    Pitch(k) = gimbalMsg{1}.Pitch;
    Yaw(k) = gimbalMsg{1}.Yaw;
    Roll(k) = gimbalMsg{1}.Roll;
end
