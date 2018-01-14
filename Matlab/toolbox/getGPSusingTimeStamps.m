%% Determine closest GPS for images
clc, clear

%% - Inputs:
% load timeColor.mat
% load seqColor.mat
% load timeIR.mat
% load seqIR.mat
% load timeGPS.mat
% load seqGPS.mat
%
% load AltitudeFalcon.mat
% load LatitudeFalcon.mat
% load LongitudeFalcon.mat

file = fopen('/media/thanu/raghavshdd1/2017-06-22/Ximea/geolocation.txt', 'wt');
% file2 = fopen('imagesColor.txt', 'wt');

bag = rosbag('/media/thanu/raghavshdd1/2017-05-18/RealsenseXimea/falcon-2017-05-18-16-46-46.bag');
imgTopic = select(bag,'Topic', '/falcon/color/color');
gpsTopic = select(bag,'Topic', '/falcon/dji_ros/gps');

%% Extract data

% Number of messages
NoI = imgTopic.NumMessages;
NoG = gpsTopic.NumMessages;

% Extract publish time
% imageTime = imgTopic.MessageList.Time;

imageTime = zeros(NoI,1);

for i = 1: NoI
    imgMsg = readMessages(imgTopic, i);
    imageTime(i) = imgMsg{1}.Header.Stamp.Sec + imgMsg{1}.Header.Stamp.Nsec / (10^9);
end

gpsLaLoAlT = zeros(NoG,4);

for j = 1:NoG
    gpsMsg = readMessages(gpsTopic, j);
    gpsLaLoAlT(j,1) = gpsMsg{1}.Latitude;
    gpsLaLoAlT(j,2) = gpsMsg{1}.Longitude;
    gpsLaLoAlT(j,3) = gpsMsg{1}.Altitude;
    gpsLaLoAlT(j,4) = gpsMsg{1}.Header.Stamp.Sec + gpsMsg{1}.Header.Stamp.Nsec / (10^9);
end
%% Find closest time stamp in GPS

% time = timeIR;
% seq = seqIR;

% Number of frames
% NoF = size(time,2);

%Save every p-th image
p = 1;

for frameIt = 1:NoI
    
    if mod(frameIt-1,p) == 0
        
        %value to find
        val = imageTime(frameIt);
        
        tmp = abs(gpsLaLoAlT(:,4)-val);
        
        %index of closest value
        [value, idx] = min(tmp);
        
        %closest value
        closest = gpsLaLoAlT(idx,4);
        
        Alt = num2str(gpsLaLoAlT(idx,3));
        Lat = num2str(gpsLaLoAlT(idx,1));
        Long = num2str(gpsLaLoAlT(idx,2));
        Seq = num2str(frameIt-1);
        
        if (frameIt-1) < 10
            fprintf(file,['frame000' Seq '.jpg,' Lat ',' Long ',' Alt '\n']);
            % fprintf(file2,['frame000' Seq '.jpg\n']);
            
        elseif (frameIt-1) < 100
            fprintf(file,['frame00' Seq '.jpg,' Lat ',' Long ',' Alt '\n']);
            % fprintf(file2,['frame00' Seq '.jpg\n']);
            
        elseif (frameIt-1) < 1000
            fprintf(file,['frame0' Seq '.jpg,' Lat ',' Long ',' Alt '\n']);
            % fprintf(file2,['frame0' Seq '.jpg\n']);
            
        elseif (frameIt-1) < 10000
            fprintf(file,['frame' Seq '.jpg,' Lat ',' Long ',' Alt '\n']);
            % fprintf(file2,['frame' Seq '.jpg\n']);
        end
    end
end

fclose(file);