function ptCloud = plyread(filename)
% Validate the input
if isstring(filename)
    filename = char(filename);
end

if ~ischar(filename)
    error(message('vision:pointcloud:badFileName'));
end

% Validate the file type
idx = find(filename == '.');
if (~isempty(idx))
    extension = lower(filename(idx(end)+1:end));
else
    extension = '';
end

% Validate the file extension.
if(~(strcmpi(extension,'pcd') || strcmpi(extension,'ply')))
    error(message('vision:pointcloud:unsupportedFileExtension'));
end

% Verify that the file exists.
fid = fopen(filename, 'r');
if (fid == -1)
    if ~isempty(dir(filename))
        error(message('MATLAB:imagesci:imread:fileReadPermission', filename));
    else
        error(message('MATLAB:imagesci:imread:fileDoesNotExist', filename));
    end
else
    % File exists.  Get full filename.
    filename = fopen(fid);
    fclose(fid);
end

if( strcmpi(extension,'ply') )
    % Read properties of 'Vertex'
    elementName = 'vertex';
    requiredProperties = {'x','y','z'};
    % Alternative names are specified in a cell array within the main cell array.
    optionalProperties = {{'red','diffuse_red'},{'green','diffuse_green'},{'blue','diffuse_blue'},'nx','ny','nz'};
    properties = visionPlyRead(filename,elementName,requiredProperties,optionalProperties);

    % Get location property
    x = properties{1};
    y = properties{2};
    z = properties{3};
    if isa(x,'double') || isa(y,'double') || isa(z,'double')
        loc = [double(x), double(y), double(z)];
    else
        loc = [single(x), single(y), single(z)];
    end

    % Get color property
    r = properties{4};
    g = properties{5};
    b = properties{6};
    color = [im2uint8(r), im2uint8(g), im2uint8(b)];

    % Get normal property
    nx = properties{7};
    ny = properties{8};
    nz = properties{9};
    if isa(nx,'double') || isa(ny,'double') || isa(nz,'double')
        normal = [double(nx), double(ny), double(nz)];
    else
        normal = [single(nx), single(ny), single(nz)];
    end
    
elseif( strcmpi(extension,'pcd') )
    requiredProperties = {'x','y','z'};
    optionalProperties = {'r','g','b','normal_x','normal_y','normal_z'};
    properties = visionPcdRead(filename,requiredProperties,optionalProperties);
    
    % Get location property
    x = properties{1};
    y = properties{2};
    z = properties{3};
    % Get color property
    r = properties{4};
    g = properties{5};
    b = properties{6}; 
    % Get normal property
    nx = properties{7};
    ny = properties{8};
    nz = properties{9};
    
    [~,cols] = size(x);
    
    % Check if it is organized or unorganized point cloud
    if cols == 1 
        dim = 2;
    else
        dim = 3;
    end
    
    if isempty(x) || isempty(y) || isempty(z)
        loc = [];
    else
        if isa(x,'double') || isa(y,'double') || isa(z,'double')
            loc = cat(dim, double(x), double(y), double(z));
        else
            loc = cat(dim,single(x), single(y), single(z));
        end         
    end
    
    if isempty(r) || isempty(g) || isempty(b)
        color = [];
    else
        color = cat(dim,im2uint8(r), im2uint8(g), im2uint8(b));
    end
    
    if isempty(nx) || isempty(ny) || isempty(nz)
        normal = [];
    else
        if isa(nx,'double') || isa(ny,'double') || isa(nz,'double')
            normal = cat(dim, double(nx), double(ny), double(nz));
        else
            normal = cat(dim,single(nx), single(ny), single(nz));
        end         
    end    
end

ptCloud = pointCloud(loc, 'Color', color, 'Normal', normal);