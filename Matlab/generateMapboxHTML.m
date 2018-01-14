
%    Created on: May 31, 2017
%    Author: Thanujan Mohanadasan
%    Institute: ETH Zurich, Autonomous Systems Lab

%    Modified on: Dec 20, 2017
%    Author: Jannic Veith
%    Institute: ETH Zurich, Autonomous Systems Lab

%% Generate html file for mapbox visualization
clc, clear

%% raghavshdd1 location
hddLoc = '/Volumes/raghavshdd1/';

%% Inputs
% - Template location:
templateFile = ['/mapboxHTML/template.htm'];

% - Dataset folder:
dataFolder = ['../Visualization/Datasets/'];

% - Output file name & location:
output = ['../Visualization/HTML/visualization.htm'];

%% Read files from folder
% Sort folders by date
folderContent = dir(dataFolder);
dates = {folderContent(3:end).name};
dates = datestr(dates);
dates = datenum(dates);
[~,ndx] = sort(dates);
folderContent = folderContent(ndx+2);

% - Number of datasets:
NoD = size(folderContent,1);

for iSet = 1:NoD
    % - Dates:
    dateList{iSet,1} = folderContent(iSet).name;
    
    % - Sequoia?
    sequoia(iSet) = 0;
    
    % - Images:
    imageDir = dir([dataFolder dateList{iSet} '/*.png']);
    [~,ndx] = natsortfiles({imageDir.name});
    imageDir = imageDir(ndx);
    % - Number of images per dataset:
    NoI(iSet) = size(imageDir,1);
    for iImage = 1:NoI(iSet)
        imageStruct(iImage,iSet).direction = [imageDir(iImage).folder '/' imageDir(iImage).name];
        [~,imageStruct(iImage,iSet).type,~] = fileparts(imageStruct(iImage,iSet).direction);
        if strcmp(imageStruct(iImage,iSet).type, 'NIR')
            sequoia(iSet) = 1;
        end
    end
    
    % - For image latitude, longitude bounds: camera parameters, wgs84, pointcloud files
    cP = dir([dataFolder dateList{iSet} '/*camera_parameters.txt']);
    cameraParam{iSet} = [cP.folder '/' cP.name];
    
    llF = dir([dataFolder dateList{iSet} '/*camera_parameters_wgs84.txt']);
    latlongFile{iSet} = [llF.folder '/' llF.name];
    
    pN = dir([dataFolder dateList{iSet} '/*.ply']);
    pcName{iSet} = [pN.folder '/' pN.name];
end

%% Calculate bounding boxes
for i = 1:NoD
    bound(i,:) = boundingboxCoordinates(cameraParam{i}, latlongFile{i}, pcName{i}, sequoia(i));
end

%% Read template
fid = fopen(templateFile,'r');
i = 1;
tline = fgetl(fid);
template{i} = tline;

while ischar(tline)
    i = i+1;
    tline = fgetl(fid);
    template{i} = tline;
end

fclose(fid);

%% Fill template
% Slider parameters; line 167
sliderLine = 167;
slider = {['<input id="slider" type="range" min="0" max="' num2str(NoD-1) '" step="1" value="0" />']};

sliderRows = size(slider,1);
template(sliderLine+(sliderRows-1):end+(sliderRows-1)) = template(sliderLine:end);
template(sliderLine:sliderLine+(sliderRows-1)) = slider;

% Add images and bounding box coordinates; line 190
coordinatesLine = 190 + sliderRows-1;
coordinates = cell(0);

for i = 1:NoD
    
    for j = 1:NoI(i)
        if i == 1 && j == 1
            coordinates(end+1) = {['var imageUrl' num2str(i) '_' num2str(j) '= "' imageStruct(j,i).direction '",']};
        else
            coordinates(end+1) = {['imageUrl' num2str(i) '_' num2str(j) '= "' imageStruct(j,i).direction '",']};
        end
        
    end
    
    if i ~= NoD
        coordinates(end+1:end+3) = {['imageBounds' num2str(i) '= L.latLngBounds(['];
            ['[' eval(['num2str(bound(' num2str(i) ',1),8)']) ', ' eval(['num2str(bound(' num2str(i) ',2),8)']) '],'];
            ['[' eval(['num2str(bound(' num2str(i) ',3),8)']) ', ' eval(['num2str(bound(' num2str(i) ',4),8)']) ']]),']};
    else
        coordinates(end+1:end+3) = {['imageBounds' num2str(i) '= L.latLngBounds(['];
            ['[' eval(['num2str(bound(' num2str(i) ',1),8)']) ', ' eval(['num2str(bound(' num2str(i) ',2),8)']) '],'];
            ['[' eval(['num2str(bound(' num2str(i) ',3),8)']) ', ' eval(['num2str(bound(' num2str(i) ',4),8)']) ']]);']};
    end
    
end

coordinates = coordinates';
coordinatesRows = size(coordinates,1);
template(coordinatesLine+(coordinatesRows-1):end+(coordinatesRows-1)) = template(coordinatesLine:end);
template(coordinatesLine:coordinatesLine+(coordinatesRows-1)) = coordinates;

% Default dataset; line 432
defaultLine = 432 + coordinatesRows + sliderRows-2;
default = cell(0);

for j = 1:NoI(1)
    default(end+1:end+2) = {['addImage(imageUrl1_' num2str(j) ',imageBounds1,link' imageStruct(j,1).type ', imageGroup);'];
        ['layers.appendChild(link' imageStruct(j,1).type ');']};
end

default(end+1:end+2) = {'hideAll(linkHIDE,imageGroup);';
    'layers.appendChild(linkHIDE);'};

default = default';
defaultRows = size(default,1);
template(defaultLine+(defaultRows-1):end+(defaultRows-1)) = template(defaultLine:end);
template(defaultLine:defaultLine+(defaultRows-1)) = default;

% Add dates to struct; line 437
dateLine = 437 + coordinatesRows + sliderRows + defaultRows-3;
date = cell(0);

for i = 1:NoD
    date(end+1) = {['"' dateList{i} '",']};
end

date = date';
dateRows = size(date,1);
template(dateLine+(dateRows-1):end+(dateRows-1)) = template(dateLine:end);
template(dateLine:dateLine+(dateRows-1)) = date;

% Switch between datasets; line: 489
switchesLine = 489 + coordinatesRows + sliderRows + defaultRows + dateRows-4;
switches = cell(0);

for i = 1:NoD
    switches(end+1:end+5) = {['if (dataSet == ' num2str(i-1) ') {'];
        'imageGroup.clearLayers();';
        'while (layers.firstChild) {';
        'layers.removeChild(layers.firstChild);';
        '}'};
    
    for j = 1:NoI(i)
        switches(end+1:end+3) = {['addImage(imageUrl' num2str(i) '_' num2str(j) ',imageBounds' num2str(i) ',link' imageStruct(j,i).type ', imageGroup);'];
            ['layers.appendChild(link' imageStruct(j,i).type ');'];
            ['link' imageStruct(j,i).type '.className = "active";']};
    end
    
    switches(end+1:end+2) = {'hideAll(linkHIDE,imageGroup);';
        'layers.appendChild(linkHIDE);'};
    
    switches(end+1) = {'}'};
end

switches = switches';
switchesRows = size(switches,1);
template(switchesLine+(switchesRows-1):end+(switchesRows-1)) = template(switchesLine:end);
template(switchesLine:switchesLine+(switchesRows-1)) = switches;

%% Write back to file
fid = fopen(output, 'w');

for i = 1:numel(template)
    
    if template{i+1} == -1
        fprintf(fid,'%s', template{i});
        break
    else
        fprintf(fid,'%s\n', template{i});
    end
    
end