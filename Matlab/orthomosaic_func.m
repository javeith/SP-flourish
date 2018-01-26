function [] = orthomosaic_func(pcName, camyaml, ximea, output)
%% Orthomosaic by projecting point cloud on a plane and filling holes
% Orthomosaic after reprojection

%% Inputs needed:
% - colored Point cloud
% pcName = '/media/thanu/raghavshdd1/Ximea_Tamron/20170613/clouds/band2.ply';
%
% % - Intrinsic parameters (.yaml)
% camyaml = '/media/thanu/raghavshdd1/Ximea_Tamron/20170613/intrinsics_ximea.yaml';
%
% % - Ximea?
% ximea = 1;
%
% % - Output name:
% output = '/media/thanu/raghavshdd1/Ximea_Tamron/20170613/Orthomosaics/band2.png';

%% Modes
% - Average of neighboring nonzero pixels -> 8 neighbors interpolation

Mode = 'Average';
NoN = 1; % Number of neighbors [(2 * NoN + 1) x (2 * NoN + 1)] (gets blurry with NoN = 2)
NoI = 5; % Number of iterations

% Display counters

disp_NoP = 1;
disp_columns = 0;
disp_NoI = 1;


% - Interpolate between nonzero pixels to fill gaps: Find

% Mode = 'Interpolation';
% NoN = 3;
% NoI = 3;

%% Read cam0 point cloud from Pix4D
cam1_PC = plyread(pcName);
disp(['Input: ' pcName '.']);
disp(['Intrinsics: ' camyaml '.']);
disp(['Output: ' output '.']);

% Number of points in cloud
NoP = size(cam1_PC.Location,1);

%% Camera matrix (intrinsics.yaml)
% Load camera matrix cam1
intrinsic1 = ReadYaml(camyaml);
cameraMatrix1(1,:) = cell2mat(intrinsic1.camera_matrix.data(1:3));
cameraMatrix1(2,:) = cell2mat(intrinsic1.camera_matrix.data(4:6));
cameraMatrix1(3,:) = cell2mat(intrinsic1.camera_matrix.data(7:9));

% Get image size
imageWidth1 = intrinsic1.image_width;
imageHeight1 = intrinsic1.image_height;

if ximea == 1
    imageWidth1 = 2048;
    imageHeight1 = 1088;
end

clear cam1Images cam1yaml filename intrinsic1 pcName srcFiles

%% Initilize Image size
point0 = cameraMatrix1 * [0;0;1];
point1 = cameraMatrix1 * [1;1;1];

% [m/pixel]
GSD_x = 1 / (point1(1) - point0(1));
GSD_y = 1 / (point1(2) - point0(2));

% Get max and min in x&y (point cloud)
max_x = max(cam1_PC.Location(:,1));
min_x = min(cam1_PC.Location(:,1));

max_y = max(cam1_PC.Location(:,2));
min_y = min(cam1_PC.Location(:,2));

range_x = max_x - min_x;
range_y = max_y - min_y;

% Calculate number of pixels of image
imageHeightMosaic = ceil(range_y / GSD_y);

% Scale
factor = ceil(imageHeightMosaic / imageHeight1);
GSD_x = GSD_x*factor;
GSD_y = GSD_y*factor;

imageWidthMosaic = ceil(range_x / GSD_x);
imageHeightMosaic = ceil(range_y / GSD_y);

pcMosaic = uint8(zeros(imageHeightMosaic, imageWidthMosaic, 3));

%% Fill pcMosaic with color (3D coordinate to pixel coordinate)
LocationMatrix = cam1_PC.Location;
colorMatrix = cam1_PC.Color;

% Only save color of highest point
min_z = min(cam1_PC.Location(:,3));
zBuffer = ones(imageHeightMosaic, imageWidthMosaic) * floor(min_z);

for point = 1:NoP
    
%     u = ceil((LocationMatrix(point,1) + norm(min_x)) / GSD_x);
%     v = ceil((max_y - LocationMatrix(point,2)) / GSD_y);
% Jannic Veith Edit: make this operation valid for all min_x, not
% only for min_x < 0 
    u = ceil((LocationMatrix(point,1) - min_x) / GSD_x);
    v = ceil((max_y - LocationMatrix(point,2)) / GSD_y);

    
    if u == 0
        u = 1;
    end
    
    if v == 0
        v = 1;
    end
    
    if LocationMatrix(point,3) > zBuffer(v,u)
        pcMosaic(v,u,:) = colorMatrix(point,:);
        zBuffer(v,u) = LocationMatrix(point,3);
    end
    
    % Display information
    if disp_NoP == 1
        if mod(point,100000) == 0
            disp(['Filling image, ' num2str(point) ' out of ' num2str(NoP) ' points']);
        end
    end
end

%% Save original image as bitmap
% imwrite(pcMosaic, 'pcMosaic.bmp');
% figure
% imshow(pcMosaic);

%% Interpolate between pixels to fill holes
red = pcMosaic(:,:,1);
green = pcMosaic(:,:,2);
blue = pcMosaic(:,:,3);

filledRed = pcMosaic(:,:,1);
filledGreen = pcMosaic(:,:,2);
filledBlue = pcMosaic(:,:,3);

% [X_red,Y_red] = meshgrid(1:size(red,2), 1:size(red,1));

u = size(pcMosaic,1);
v = size(pcMosaic,2);

for itIt = 1:NoI
    
    % Calculate distance to next non-zero pixel and index of it
    [D_red,~] = bwdist(red);
    [D_green,~] = bwdist(green);
    [D_blue,~] = bwdist(blue);
    
    for uIt = 1:u
        
        for vIt = 1:v
            
            % Check if "hole"-pixel has non-zero neighbors
            if (D_red(uIt, vIt) > 0 && D_red(uIt, vIt) <= sqrt(2*NoN^2)) && ...
                    (D_green(uIt, vIt) > 0 && D_green(uIt, vIt) <= sqrt(2*NoN^2)) && ...
                    (D_blue(uIt, vIt) > 0 && D_blue(uIt, vIt) <= sqrt(2*NoN^2))
                neighborsRed = zeros((2*NoN)+1);
                neighborsGreen = zeros((2*NoN)+1);
                neighborsBlue = zeros((2*NoN)+1);
                
                if strcmp(Mode,'Average')
                    % Determine neighbors of point
                    neighborsRed(max(NoN+2-uIt, 1):min(-uIt+u+NoN+1, 2*NoN+1), max(NoN+2-vIt, 1):min(-vIt+v+NoN+1, 2*NoN+1)) ...
                        = red(max((uIt-NoN), 1):min((uIt+NoN), u), max((vIt-NoN), 1):min((vIt+NoN), v));
                    
                    neighborsGreen(max(NoN+2-uIt, 1):min(-uIt+u+NoN+1, 2*NoN+1), max(NoN+2-vIt, 1):min(-vIt+v+NoN+1, 2*NoN+1)) ...
                        = green(max((uIt-NoN), 1):min((uIt+NoN), u), max((vIt-NoN), 1):min((vIt+NoN), v));
                    
                    neighborsBlue(max(NoN+2-uIt, 1):min(-uIt+u+NoN+1, 2*NoN+1), max(NoN+2-vIt, 1):min(-vIt+v+NoN+1, 2*NoN+1)) ...
                        = blue(max((uIt-NoN), 1):min((uIt+NoN), u), max((vIt-NoN), 1):min((vIt+NoN), v));
                    
                    % Number of non-zero elements
                    NoE_red = size(find(neighborsRed ~= 0),1);
                    NoE_green = size(find(neighborsGreen ~= 0),1);
                    NoE_blue = size(find(neighborsBlue ~= 0),1);
                    
                    % Take mean of all non-zero elements
                    filledRed(uIt, vIt) = sum(neighborsRed(:))/NoE_red;
                    filledGreen(uIt, vIt) = sum(neighborsGreen(:))/NoE_green;
                    filledBlue(uIt, vIt) = sum(neighborsBlue(:))/NoE_blue;
                end
                
                if strcmp(Mode,'Interpolation')
                    % Determine neighbors of point
                    neighborsRed(max(NoN+2-uIt, 1):min(-uIt+u+NoN+1, 2*NoN+1), max(NoN+2-vIt, 1):min(-vIt+v+NoN+1, 2*NoN+1)) ...
                        = red(max((uIt-NoN), 1):min((uIt+NoN), u), max((vIt-NoN), 1):min((vIt+NoN), v));
                    
                    neighborsGreen(max(NoN+2-uIt, 1):min(-uIt+u+NoN+1, 2*NoN+1), max(NoN+2-vIt, 1):min(-vIt+v+NoN+1, 2*NoN+1)) ...
                        = green(max((uIt-NoN), 1):min((uIt+NoN), u), max((vIt-NoN), 1):min((vIt+NoN), v));
                    
                    neighborsBlue(max(NoN+2-uIt, 1):min(-uIt+u+NoN+1, 2*NoN+1), max(NoN+2-vIt, 1):min(-vIt+v+NoN+1, 2*NoN+1)) ...
                        = blue(max((uIt-NoN), 1):min((uIt+NoN), u), max((vIt-NoN), 1):min((vIt+NoN), v));
                    
                    % Find boundaries for interpolation
                    west = find(neighborsRed(NoN+1,1:NoN),1,'last');
                    east = find(neighborsRed(NoN+1,(NoN+2):end),1);
                    north = find(neighborsRed(1:NoN,NoN+1),1,'last');
                    south = find(neighborsRed((NoN+2):end,NoN+1),1);
                    
                    % Row
                    if ~isempty(west) && ~isempty(east)
                        
                        % Linear interpolation between boundaries
                        intRed = neighborsRed(NoN+1,west) + ((neighborsRed(NoN+1,NoN+1+east) ...
                            - neighborsRed(NoN+1,west)) / (NoN+1+east-west) * (NoN+1-west));
                        
                        intGreen = neighborsGreen(NoN+1,west) + ((neighborsGreen(NoN+1,NoN+1+east) ...
                            - neighborsGreen(NoN+1,west)) / (NoN+1+east-west) * (NoN+1-west));
                        
                        intBlue = neighborsBlue(NoN+1,west) + ((neighborsBlue(NoN+1,NoN+1+east) ...
                            - neighborsBlue(NoN+1,west)) / (NoN+1+east-west) * (NoN+1-west));
                        
                        filledRed(uIt, vIt) = ceil(intRed);
                        filledGreen(uIt, vIt) = ceil(intGreen);
                        filledBlue(uIt, vIt) = ceil(intBlue);
                    end
                    
                    % Column
                    if ~isempty(north) && ~isempty(south)
                        intRed = neighborsRed(north,NoN+1) + (neighborsRed(NoN+1+south,NoN+1) - ...
                            neighborsRed(north,NoN+1)) / ((NoN+1+south-north) * (NoN+1-north));
                        
                        intGreen = neighborsGreen(north,NoN+1) + (neighborsGreen(NoN+1+south,NoN+1) - ...
                            neighborsGreen(north,NoN+1)) / ((NoN+1+south-north) * (NoN+1-north));
                        
                        intBlue = neighborsBlue(north,NoN+1) + (neighborsBlue(NoN+1+south,NoN+1) - ...
                            neighborsBlue(north,NoN+1)) / ((NoN+1+south-north) * (NoN+1-north));
                        
                        % Bilinear
                        if filledRed(uIt, vIt) ~= 0
                            filledRed(uIt, vIt) = ceil((double((filledRed(uIt, vIt))) + intRed)/2);
                            filledGreen(uIt, vIt) = ceil((double((filledGreen(uIt, vIt))) + intGreen)/2);
                            filledBlue(uIt, vIt) = ceil((double((filledBlue(uIt, vIt))) + intBlue)/2);
                            
                        else
                            filledRed(uIt, vIt) = ceil(intRed);
                            filledGreen(uIt, vIt) = ceil(intGreen);
                            filledBlue(uIt, vIt) = ceil(intBlue);
                            
                        end
                        
                    end
                    
                end
                
            end
            
        end
        
        if disp_columns == 1
            if mod(uIt,100) == 0
                disp(['Interpolating images, ' num2str(uIt) ' out of ' num2str(u) ' imagecolumns']);
            end
        end
        
    end
    
    % Prepare for next iteration
    red = filledRed;
    green = filledGreen;
    blue = filledBlue;
    
    %Display Iterations
    if disp_NoI == 1
        disp(['Completed ' num2str(itIt) ' out of ' num2str(NoI) ' iterations of Interpolation.']);
    end
end

% Assign back to pcMosaic
pcMosaic(:,:,1) = filledRed;
pcMosaic(:,:,2) = filledGreen;
pcMosaic(:,:,3) = filledBlue;

clear red green blue colorMatrix LocationMatrix D_blue D_red D_green L_blue L_red L_green ...
    factor filledRed filledGreen filledBlue itIt max_x max_y max_z min_x min_y min_z ...
    neighborsRed neighborsGreen neighborsBlue NoE point0 point1 north south west east ...
    intRed intGreen intBlue uIt vIt

%% Save filled image
imwrite(pcMosaic, output,'Transparency',[0 0 0]);
% figure
% imshow(pcMosaic);

end