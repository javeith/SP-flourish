function [ extParam ] = readExternalParametersFile( cameraParam, sequoia, initial_channel )
%This function is used to read the external_parameters.txt from Pix4D

delimiter = ' ';
if ~strcmp(initial_channel,'rgb') && sequoia == 1
    startRow = 4;
else
    startRow = 8;
end

formatSpec = '%s%s%s%[^\n\r]';

fileID = fopen(cameraParam,'r');

textscan(fileID, '%[^\n\r]', startRow-1, 'WhiteSpace', '', 'ReturnOnError', false, 'EndOfLine', '\r\n');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'ReturnOnError', false);

fclose(fileID);

% Replace non-numeric text with NaN.
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = dataArray{col};
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,2,3]
    % Converts text in the input cell array to numbers. Replaced non-numeric
    % text with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1)
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            if sequoia == 1
                if strcmp(initial_channel,'rgb') && mod(row,10) == 1
                    result = sscanf(rawData{row},'IMG_%d_%d_%d');
                    numbers = num2str(result(3));
                elseif ~strcmp(initial_channel,'rgb') && mod(row,5) == 1
                    result = sscanf(rawData{row},'IMG_%d_%d_%d');
                    numbers = num2str(result(3));
                else
                    result = regexp(rawData{row}, regexstr, 'names');
                    numbers = result.numbers;
                end
            else
                result = regexp(rawData{row}, regexstr, 'names');
                numbers = result.numbers;
            end
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if any(numbers==',')
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'))
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric text to numbers.
            if ~invalidThousandsSeparator
                numbers = textscan(strrep(numbers, ',', ''), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch me
        end
    end
end

R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

extParam(:,1) = cell2mat(raw(:, 1));
extParam(:,2) = cell2mat(raw(:, 2));
extParam(:,3) = cell2mat(raw(:, 3));


end

