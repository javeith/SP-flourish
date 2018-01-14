function boundingBox = boundingboxCoordinates(cameraParam, latlongFile, pcName, sequoia)
%% Get gps coordinates of the bounding box of a orthomosaic

%% Inputs:

% - Frame parameters from Pix4D (_calibrated_camera_parameters.txt):
% Includes pose of each camera
% cameraParam = '/data/2017-04-13/20170413-sugarbeet-sequoia-eschikon-flight2_calibrated_camera_parameters.txt';
% cameraParam = '/data/2016-10-13/flourish3_calibrated_camera_parameters.txt';
% cameraParam = '/data/2017-05-05/20170505-sugarbeet-sequoia-rgb-eschikon-n2_calibrated_camera_parameters.txt';

% - Latitude, Longitude file from Pix4D for each camera (_calibrated_external_camera_parameters_wgs84.txt):
% latlongFile = '/data/2017-04-13/20170413-sugarbeet-sequoia-eschikon-flight2_calibrated_external_camera_parameters_wgs84.txt';
% latlongFile = '/data/2016-10-13/flourish3_calibrated_external_camera_parameters_wgs84.txt';
% latlongFile = '/data/2017-05-05/20170505-sugarbeet-sequoia-rgb-eschikon-n2_calibrated_external_camera_parameters_wgs84.txt';

% - Point cloud
% pcName = '/data/2017-04-13/20170413-sugarbeet-sequoia-eschikon-flight2_group1_densified_point_cloud.ply';
% pcName = '/data/2016-10-13/flourish3_group1_densified_point_cloud.ply';
% pcName = '/data/2017-05-05/downsampled_PC.ply';

% - Sequoia?
% sequoia = 1;        % [0, 1]

%% T from external_parameters.txt (from Pix4D)
[ extParam ] = readExternalParametersFile( cameraParam, sequoia, 'rgb' );

NoLextParam = size(extParam,1);
w2cIt = 1;
cameraT = zeros(NoLextParam,4);



while w2cIt <= NoLextParam
    cameraT(w2cIt,1) = extParam(w2cIt,1);
    cameraT(w2cIt,2:4) = extParam(w2cIt+6,:);
    
    w2cIt = w2cIt + 10;
end

cameraT( ~any(cameraT,2), : ) = [];

%% Get latitude & longitude for each camera
[ longlatAlt ] = readLatLongParametersFile( latlongFile );

%% Read cam0 point cloud from Pix4D
cam0_PC = plyread(pcName);
LocationMatrix = cam0_PC.Location;

%% Convert longitude & latitude to x,y,z in wgs84
wgs84 = wgs84Ellipsoid('meters');

% Optimize
err_func = @(d) getErr(cameraT,longlatAlt,d);
options = optimoptions('fminunc','MaxFunctionEvaluations', 500);
[D,err] = fminunc(err_func, [0,0,0],options);

lat0 = D(1);
lon0 = D(2);
h0 = D(3);


[lat,lon,h] = ned2geodetic(LocationMatrix(:,2),LocationMatrix(:,1),LocationMatrix(:,3),lat0,lon0,h0,wgs84);

boundLat(1) = min(lat);
boundLat(2) = max(lat);
boundLon(1) = min(lon);
boundLon(2) = max(lon);

boundingBox = [boundLat(2), boundLon(2), boundLat(1), boundLon(1)];

%% Functions
    function [ err ] = getErr(cameraT,longlatAlt, D)
        
        lat0 = D(1);
        lon0 = D(2);
        h0 = D(3);
        
        wgs84 = wgs84Ellipsoid('meters');
        [xNorth,yEast,zDown] = geodetic2ned(longlatAlt(:,2),longlatAlt(:,1),longlatAlt(:,3),lat0,lon0,h0,wgs84);
        
        %Calculate error
        errX = mean(xNorth - cameraT(:,2));
        errY = mean(yEast - cameraT(:,3));
        errZ = mean(zDown - cameraT(:,4));
        
        err = sqrt(errX^2 + errY^2 + errZ^2);
        
    end

end