%% Extract needed information from NIR25_metadata.txt
clc, clear

%% raghavshdd1 location
hddLoc = '/media/thanu/raghavshdd1/';

%% Inputs:
% - Location of NIR25_metadata.txt:
fileLoc = [hddLoc 'thanujan/Datasets/FIP/20170622/NIR25/NIR25_metadata.txt'];

% Image location:
imgLoc = [hddLoc 'thanujan/Datasets/FIP/20170622/NIR25/'];

% - Save location:
saveLoc = [hddLoc 'thanujan/Datasets/FIP/20170622/geolocation.txt'];

%% Read file
% Initialize variables
delimiter = ',';

% Read columns of data as text
formatSpec = '%*s%*s%s%*s%*s%*s%*s%s%s%s%[^\n\r]';

% Open the text file
fileID = fopen(fileLoc,'r');

% Read columns of data according to the format
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);

% Close the text file
fclose(fileID);

% Convert the contents of columns containing numeric text to numbers
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = dataArray{col};
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[2,3,4]
    rawData = dataArray{col};
    for row=1:size(rawData, 1);
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData{row}, regexstr, 'names');
            numbers = result.numbers;
            
            invalidThousandsSeparator = false;
            if any(numbers==',');
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'));
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            if ~invalidThousandsSeparator;
                numbers = textscan(strrep(numbers, ',', ''), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch me
        end
    end
end

% Split data into numeric and cell columns
rawNumericColumns = raw(:, [2,3,4]);
rawCellColumns = raw(:, 1);

% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),rawNumericColumns); % Find non-numeric cells
rawNumericColumns(R) = {NaN}; % Replace non-numeric cells

% Allocate imported array to column variable names
fileLoc = rawCellColumns(:, 1);
lat = cell2mat(rawNumericColumns(:, 1));
lon = cell2mat(rawNumericColumns(:, 2));
alt = cell2mat(rawNumericColumns(:, 3));


% Clear temporary variables
clearvars delimiter formatSpec fileID dataArray ans raw col numericData rawData row regexstr ...
    result numbers invalidThousandsSeparator thousandsRegExp me rawNumericColumns rawCellColumns R;

%% Remove lines without image
j=1;
for i = 2:size(alt,1)
    if ~exist([imgLoc char(fileLoc(i))], 'file')
        toDelete(j) = i;
        j = j+1;
    end
end

if exist('toDelete')
    fileLoc(toDelete,:) = [];
    alt(toDelete,:) = [];
    lat(toDelete,:) = [];
    lon(toDelete,:) = [];
end

%% Save output
file = fopen(saveLoc, 'wt');

for i = 2:size(alt,1)
    fileID = i-2;
    if (fileID) < 10
        fprintf(file,['frame000' num2str(fileID) '.tif' ',' num2str(lat(i),8) ',' num2str(lon(i),8) ',' num2str(alt(i),8) '\n']);
        
    elseif (fileID) < 100
        fprintf(file,['frame00' num2str(fileID) '.tif' ',' num2str(lat(i),8) ',' num2str(lon(i),8) ',' num2str(alt(i),8) '\n']);
        
    elseif (fileID) < 1000
        fprintf(file,['frame0' num2str(fileID) '.tif' ',' num2str(lat(i),8) ',' num2str(lon(i),8) ',' num2str(alt(i),8) '\n']);
        
    elseif (fileID) < 10000
        fprintf(file,['frame' num2str(fileID) '.tif' ',' num2str(lat(i),8) ',' num2str(lon(i),8) ',' num2str(alt(i),8) '\n']);
        
    end
end

fclose(file);