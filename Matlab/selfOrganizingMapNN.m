%% Self-Organizing Map Neural Network
% Self-organizing maps (SOMs) are very good at creating classifications.
% Further, the classifications retain topological information about which
% classes are most similar to others.  Self-organizing maps can be created
% with any desired level of detail.  They are particularly well suited for
% clustering data in many dimensions and with complexly shaped and
% connected feature spaces.
clear, clc

%% raghavshdd1 location
hddLoc = '/media/thanu/raghavshdd1/';

%% Inputs:
% Training set:
NIRLoc = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170622/clouds/'];
VISLoc = [hddLoc 'thanujan/Datasets/Ximea_Tamron/20170622/VIS_clouds/'];

% Save location for point cloud:
output = [hddLoc 'thanujan/Datasets/xClassifier/x41bands/SOM/3x3/SOM_3x3_trainedSet_170622_OWNCOLORS.ply'];

%% Read point clouds & extract data for net
for iBand = 1:25
    pc = plyread([NIRLoc 'band'  num2str(iBand) '.ply']);
    x(iBand,:) = double(pc.Color(:,1));
    clear pc;
end

for iBand = 1:16
    pc = plyread([VISLoc 'band'  num2str(iBand) '.ply']);
    x(iBand+25,:) = double(pc.Color(:,1));
    clear pc;
end

% autoenc = trainAutoencoder(x,'useGPU',1); % NOT POSSIBLE ON CPU
% Autoencoder does not need the specification of the size of the map
%% Clustering with a Neural Network
% The next step is to create a neural network that will learn to cluster.
%
% *selforgmap* creates self-organizing maps for classify samples with as
% much detailed as desired by selecting the number of neurons in each
% dimension of the layer.

net = selforgmap([4 4]);
view(net)

%% Training
[net,tr] = train(net,x);
nntraintool

%% Class vectors
% Here the self-organizing map is used to compute the class vectors of
% each of the training inputs.

y = net(x);
cluster_index = vec2ind(y);

%% Topology
% *plotsomtop* plots the self-organizing maps topology of Y neurons
% positioned in an AxB hexagonal grid.  Each neuron has learned to
% represent a different class, with adjecent neurons typically
% representing similar classes.

plotsomtop(net)

%% Members per class
plotsomhits(net,x)
set(gca,'fontsize',18);

%% Neighbor connections
% *plotsomnc* shows the neuron neighbor connections.  Neighbors typically
% classify similar samples.

plotsomnc(net)
set(gca,'fontsize',18);

%% Neighbor distance
% *plotsomnd* shows how distant (in terms of Euclidian distance) each
% neuron's class is from its neighbors.  Connections which are bright
% indicate highly connected areas of the input space.  While dark
% connections indicate classes representing regions of the feature space
% which are far apart.

plotsomnd(net)
set(gca,'fontsize',18);

%% Input weights
% *plotsomplanes* shows a weight plane for each of the input features.
% They are visualizations of the weights that connect each input to each
% of the X neurons in the nxn hexagonal grid.  Darker colors represent
% larger weights.  If two inputs have similar weight planes (their color
% gradients may be the same or in reverse) it indicates they are highly
% correlated.

plotsomplanes(net)

%% Create colored point cloud
colorMatrix = zeros(size(cluster_index,2),3);

for iClass = 1:size(y,1)
    %    color = ceil([rand*255, rand*255, rand*255]);
    colorMatrix(cluster_index == iClass,:) = repmat(colorMapSOM(iClass,:),[size(colorMatrix(cluster_index == iClass),2),1]);
end

pc = plyread([NIRLoc 'band'  num2str(8) '.ply']);
resultCloud = pointCloud(pc.Location,'Color',uint8(colorMatrix));

pcwrite(resultCloud,output,'PLYFormat','binary');

%% Colormap
for iClass = 1:16
    colorMapSOM(iClass,:) = ceil([rand*255, rand*255, rand*255]);
end

%% Own colormap 16
colorMapSOM(1,:) = [255,0,0];
colorMapSOM(2,:) = [204,0,102];
colorMapSOM(3,:) = [206,10,10];
colorMapSOM(4,:) = [0,255,0];
colorMapSOM(5,:) = [100,100,100];
colorMapSOM(6,:) = [51,25,0];
colorMapSOM(7,:) = [0,0,255];
colorMapSOM(8,:) = [0,127,0];
colorMapSOM(9,:) = [255,128,0];
colorMapSOM(10,:) = [178,89,0];
colorMapSOM(11,:) = [0,255,0];
colorMapSOM(12,:) = [90,255,90];
colorMapSOM(13,:) = [255,156,57];
colorMapSOM(14,:) = [164,111,58];
colorMapSOM(15,:) = [67,39,12];
colorMapSOM(16,:) = [255,102,255];

%% Own colormap 9
colorMapSOM(1,:) = [255,128,0];
colorMapSOM(2,:) = [51,25,0];
colorMapSOM(3,:) = [255,0,0];
colorMapSOM(4,:) = [164,111,58];
colorMapSOM(5,:) = [67,39,12];
colorMapSOM(6,:) = [206,10,10];
colorMapSOM(7,:) = [0,255,0];
colorMapSOM(8,:) = [0,127,0];
colorMapSOM(9,:) = [255,255,0];