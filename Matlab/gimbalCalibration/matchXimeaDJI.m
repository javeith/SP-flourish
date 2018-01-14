%% Match DJI images with ximea images using gimbal angles information
clc, clear, close all

%% Inputs

% - DJI image location:
imgLocation = '/home/thanu/Documents/CWG-CALTag2/shortBag/DJI_173553/';

% - Gimbal angles from rosbag:
% load /media/thanu/689257259256F752/CalibrationWithGimbal/Pitch.mat;
% load /media/thanu/689257259256F752/CalibrationWithGimbal/Roll.mat;
% load /media/thanu/689257259256F752/CalibrationWithGimbal/Yaw.mat;
load /home/thanu/Documents/CWG-CALTag2/shortBag/gimbalPYRT.mat;

% - Camera poses DJI and ximea:
load /home/thanu/Documents/CWG-CALTag2/shortBag/DJIPoses.mat
load /home/thanu/Documents/CWG-CALTag2/shortBag/ximeaPoses.mat

% - Camera orientation and location in world frame DJI and ximea:
load /home/thanu/Documents/CWG-CALTag2/shortBag/DJIOrientLoc.mat
load /home/thanu/Documents/CWG-CALTag2/shortBag/ximeaOrientLoc.mat

% Get poses from cameraParams:
% load /media/thanu/689257259256F752/CWG2/DJIIntrinsics.mat
% load /media/thanu/689257259256F752/CWG2/ximeaIntrinsics.mat

load /home/thanu/Documents/CWG-CALTag2/shortBag/worldPoints.mat
load /home/thanu/Documents/CWG-CALTag2/shortBag/ximeaTime.mat
load /home/thanu/Documents/CWG-CALTag2/shortBag/angleIDtoEachXimea.mat

startTime = 1497972953.64; % = 17:35:53.64
endTime = 1497973069.01;

startXimeaPoses = 186;
endXimeaPoses = 251;

%% Read gimbal angles from image
srcFiles = dir([imgLocation 'DJI_*.JPG']);

yawEXIF = zeros(size(srcFiles,1),1);
pitchEXIF = zeros(size(srcFiles,1),1);
rollEXIF = zeros(size(srcFiles,1),1);
% timeEXIF = zeros(size(srcFiles,1),1);
timeDJI = cell(size(srcFiles,1),1);

for i = 1 : (size(srcFiles,1))
    filename = strcat(imgLocation,srcFiles(i).name);
    exif = getexif(filename);
    exif = strsplit(exif);
    yawEXIF(i) = find(contains(exif, 'GimbalYawDegree'));
    yawEXIF(i) = str2num(exif{yawEXIF(i)+2});
    rollEXIF(i) = find(contains(exif, 'GimbalRollDegree'));
    rollEXIF(i) = str2num(exif{rollEXIF(i)+2});
    pitchEXIF(i) = find(contains(exif, 'GimbalPitchDegree'));
    pitchEXIF(i) = str2num(exif{pitchEXIF(i)+2});
    timeEXIF(i) = find(contains(exif, 'DateTimeOriginal'));
    timeDJI(i) = cellstr(exif{timeEXIF(i)+3});
end

%% Find corresponding from rosbag gimbal angle recordings
matchDJIRosbag = zeros(size(srcFiles,1), 3);
previous = 1;

for i = 1: size(srcFiles,1)
    % value to find
    val1 = yawEXIF(i);
    val2 = rollEXIF(i);
    val3 = pitchEXIF(i);
    
    tmp1 = abs(gimbalPYRT((previous+280):(previous+500),2)-val1);
    tmp2 = abs(gimbalPYRT((previous+280):(previous+500),3)-val2);
    tmp3 = abs(gimbalPYRT((previous+280):(previous+500),1)-val3);
    tmp = tmp1+tmp2+tmp3;
    
    % index of closest value
    [value, idx] = min(tmp);
    
    % closest index
    matchDJIRosbag(i,1) = i;
    matchDJIRosbag(i,2) = idx + previous + 280;
    matchDJIRosbag(i,3) = value;
    
    previous = matchDJIRosbag(i,2);
end

%%
% diff = zeros(size(gimbalPYRT,1),1);
% for i = 1:(size(gimbalPYRT,1)-1)
%     if i == 1
%         diff(i) = gimbalPYRT(i+1,4) -gimbalPYRT(i,4);
%     else
%         diff(i) = gimbalPYRT(i+1,4) -gimbalPYRT(i,4) + diff(i-1);
%     end
% end

%% Find corresponding by finding closest poses
% srcFiles = dir([imgLocation 'DJI_*.JPG']);
% matchDJIRosbag = zeros(size(srcFiles,1), 3);
% tmp = zeros(size(ximeaPoses,3),1);
% 
% for i = 1:size(srcFiles,1)
%     
%     for k = 1:size(ximeaPoses,3)
%         tmp(k) = sum(sum(abs(DJIPoses(1:3,4,i) - ximeaPoses(1:3,4,k))));
%     end
%     
%     [value, idx] = min(tmp);
%     
%     % closest index
%     matchDJIRosbag(i,1) = i;
%     matchDJIRosbag(i,2) = idx;
%     matchDJIRosbag(i,3) = value;
% end

%% From cameraParams structs
% ximeaCand = size(ximeacameraParams.TranslationVectors,1);
% DJICand = size(DJIcameraParams.TranslationVectors,1);
% 
% for i = 1:ximeaCand
%     ximeaPoses(:,:,i) = [ximeacameraParams.RotationMatrices(:,:,i), ximeacameraParams.TranslationVectors(i,:)';...
%         0,0,0,1];
% end
% 
% for i = 1:DJICand
%     DJIPoses(:,:,i) = [DJIcameraParams.RotationMatrices(:,:,i), DJIcameraParams.TranslationVectors(i,:)';...
%         0,0,0,1];
% end

% tmp = zeros(ximeaCand,1);
% 
% for i = 1:DJICand
%     
%     for k = 1:ximeaCand
%         tmp(k) = norm(DJIPoses(1:3,4,i) - ximeaPoses(1:3,4,k));
%     end
%     
%     [value, idx] = min(tmp);
%     
%     % closest index
%     matchDJIRosbag(i,1) = i;
%     matchDJIRosbag(i,2) = idx;
%     matchDJIRosbag(i,3) = value;
% end

%% Fill ximeaPoses & ximeaOrientLoc estimations with interpolation
for i = 1:size(ximeaPoses,3)
    if ximeaPoses(1,1,i) == 0
       if  ximeaPoses(1,1,i-1) ~= 0 && ximeaPoses(1,1,i+1) ~= 0
          ximeaPoses(:,:,i) = (ximeaPoses(:,:,i-1) + ximeaPoses(:,:,i+1)) ./ 2; 
       end
    end
end

for i = 1:size(ximeaOrientLoc,3)
    if ximeaOrientLoc(1,1,i) == 0
       if  ximeaOrientLoc(1,1,i-1) ~= 0 && ximeaOrientLoc(1,1,i+1) ~= 0
          ximeaOrientLoc(:,:,i) = (ximeaOrientLoc(:,:,i-1) + ximeaOrientLoc(:,:,i+1)) ./ 2; 
       end
    end
end

%% Plot trajectories
startDJI = 53.64+(ximeaTime(startXimeaPoses)-startTime)-60
endDJI = 53.64+(ximeaTime(endXimeaPoses)-startTime)-60
% --> DJI: 13-17

for i = 1:size(DJIOrientLoc,3)
    xDJI(i) = DJIOrientLoc(1,4,i);
    yDJI(i) = DJIOrientLoc(2,4,i);
    zDJI(i) = DJIOrientLoc(3,4,i);
end

% xDJI(xDJI == 0) = [];
% yDJI(yDJI == 0) = [];
% zDJI(zDJI == 0) = [];

for i = 1:size(ximeaOrientLoc,3)
    xXimea(i) = ximeaOrientLoc(1,4,i);
    yXimea(i) = ximeaOrientLoc(2,4,i);
    zXimea(i) = ximeaOrientLoc(3,4,i);
end

% xXimea(xXimea == 0) = [];
% yXimea(yXimea == 0) = [];
% zXimea(zXimea == 0) = [];

figure(1)
plot3(xDJI,yDJI,zDJI,'Marker','o','MarkerEdgeColor','r','LineStyle','none');
% plot(xDJI,yDJI,'Marker','o','MarkerEdgeColor','r','LineStyle','none');
grid on;
hold on;
pcshow([worldPoints,zeros(size(worldPoints,1),1)], ...
  'VerticalAxisDir','down','MarkerSize',40);
% fnplt(cscvn([xDJI;yDJI]),'b',2);
fnplt(cscvn([xDJI;yDJI;zDJI]),'b',1.5);
title('DJI trajectory');
hold off;

figure(2)
pcshow([worldPoints,zeros(size(worldPoints,1),1)], ...
  'VerticalAxisDir','down','MarkerSize',40);
grid on;
hold on;
plot3(xXimea(startXimeaPoses:endXimeaPoses),yXimea(startXimeaPoses:endXimeaPoses),zXimea(startXimeaPoses:endXimeaPoses),'Color','g','Marker','o','MarkerEdgeColor','k','LineStyle','none');
fnplt(cscvn([xXimea(startXimeaPoses:endXimeaPoses);yXimea(startXimeaPoses:endXimeaPoses);zXimea(startXimeaPoses:endXimeaPoses)]),'g',1.5);
plot3(xDJI(14:20),yDJI(14:20),zDJI(14:20),'Marker','o','MarkerEdgeColor','r','LineStyle','none');
fnplt(cscvn([xDJI(14:20);yDJI(14:20);zDJI(14:20)]),'b',1.5);
title('Ximea & DJI trajectories');
legend('CALTag','Ximea poses', 'Ximea trajectory', 'DJI poses', 'DJI trajectory');
hold off;


%% Find corresponding by finding closest gimbal angle
for i = 14:20
    
    tmp = abs(matchDJIRosbag(i,2)-correspondingGimbalAngles(startXimeaPoses:endXimeaPoses,2));
    
    [value, idx] = min(tmp);
    
    % closest index
    matchDJIXimea(i-13,1) = i;
    matchDJIXimea(i-13,2) = matchDJIRosbag(i,2);
    matchDJIXimea(i-13,3) = idx + startXimeaPoses - 1;
    matchDJIXimea(i-13,4) = correspondingGimbalAngles(matchDJIXimea(i-13,3),2);
    matchDJIXimea(i-13,5) = value;
end

%% Plot matches
figure(3)
pcshow([worldPoints,zeros(size(worldPoints,1),1)], ...
  'VerticalAxisDir','down','MarkerSize',40);
grid on;
hold on;
plot3(xXimea(matchDJIXimea(2:6,3)),yXimea(matchDJIXimea(2:6,3)),zXimea(matchDJIXimea(2:6,3)),'Color','g','Marker','o','MarkerEdgeColor','k','LineStyle','none');
fnplt(cscvn([xXimea(matchDJIXimea(2:6,3));yXimea(matchDJIXimea(2:6,3));zXimea(matchDJIXimea(2:6,3))]),'g',1.5);
plot3(xDJI(15:19),yDJI(15:19),zDJI(15:19),'Marker','o','MarkerEdgeColor','r','LineStyle','none');
fnplt(cscvn([xDJI(15:19);yDJI(15:19);zDJI(15:19)]),'b',1.5);
for i = 1:5
    plotCamera('Location',DJIOrientLoc(1:3,4,matchDJIXimea(i+1,1)),'Orientation',DJIOrientLoc(1:3,1:3,matchDJIXimea(i+1,1)),'Size',10,'Label',['D',num2str(i)]);
    plotCamera('Location',ximeaOrientLoc(1:3,4,matchDJIXimea(i+1,3)),'Orientation',ximeaOrientLoc(1:3,1:3,matchDJIXimea(i+1,3)),'Size',10,'Label',['X',num2str(i)],'Color',[0.8,0.2,0]);
end
title('Ximea & DJI trajectories: Only image pairs');
legend('CALTag','Ximea poses', 'Ximea trajectory', 'DJI poses', 'DJI trajectory');
hold off;

% 15-194; 16-206; 17-218; 18-230; 19-246

%%
% [Y, M, D, H, MN, S] = datevec(timeDJI(1));
% H*3600+MN*60+S

% tXimea = ximeaTime((startXimeaPoses):endXimeaPoses) - ximeaTime(startXimeaPoses);
% tDJI = [0:2:10]';
% t=[0:0.1:(ximeaTime(endXimeaPoses)-ximeaTime(startXimeaPoses))]';
% xXimeai = interp1(tXimea, xXimea((startXimeaPoses):endXimeaPoses)',t);
% yXimeai = interp1(tXimea, yXimea((startXimeaPoses):endXimeaPoses)',t);
% zXimeai = interp1(tXimea, zXimea((startXimeaPoses):endXimeaPoses)',t);
% xDJIi = interp1(tDJI,xDJI(15:20)',t);
% yDJIi = interp1(tDJI,yDJI(15:20)',t);
% zDJIi = interp1(tDJI,zDJI(15:20)',t);
% 
% devx=xXimeai-xDJIi;
% devy=yXimeai-yDJIi;
% devz=zXimeai-zDJIi; 

%%
% DJIAngle = rotm2eul(DJIPoses(1:3,1:3,1)).*180/pi
% ximeaAngle = DJIAngle - [yawEXIF(1),pitchEXIF(1),rollEXIF(1)]
% realXimeaAngle = rotm2eul(ximeaPoses(1:3,1:3,19)).*180/pi
% DJIAngle-realXimeaAngle

%% Save relative gimbal angles to each ximea image
YPRXimea = zeros(size(correspondingGimbalAngles,1),4);

for i = 1:size(correspondingGimbalAngles,1)
    YPRXimea(i,1) = i;
    YPRXimea(i,2) = gimbalPYRT(correspondingGimbalAngles(i,2),2) - gimbalPYRT(1,2);
    YPRXimea(i,3) = gimbalPYRT(correspondingGimbalAngles(i,2),1) - gimbalPYRT(1,1);
    YPRXimea(i,4) = gimbalPYRT(correspondingGimbalAngles(i,2),3) - gimbalPYRT(1,3);    
end

% save('YPRXimea')

%% Save relative gimbal angles to each DJI image
YPRDJI = zeros(size(srcFiles,1),4);

for i = 1:size(srcFiles,1)
    YPRDJI(i,1) = i;
    YPRDJI(i,2) = yawEXIF(i);% - yawEXIF(1);
    YPRDJI(i,3) = pitchEXIF(i);% - pitchEXIF(1);
    YPRDJI(i,4) = rollEXIF(i);% - rollEXIF(1);    
end

% save('YPRDJI')