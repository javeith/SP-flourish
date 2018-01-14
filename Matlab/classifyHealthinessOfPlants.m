
%    Created on: July 17, 2017
%    Author: Thanujan Mohanadasan
%    Institute: ETH Zurich, Autonomous Systems Lab

%% Healthy Plant Classification
clc, clear

%% raghavshdd1 location
hddLoc = '/media/thanu/raghavshdd1/';

%% Inputs:
% TRAINING 01: Mavic -> just RGB (different plants...)
% load '/media/thanu/raghavshdd1/Training01_170713_DJI_RGB/healthy.mat'
% load '/media/thanu/raghavshdd1/Training01_170713_DJI_RGB/unhealthy.mat'

% TRAINING 02: DJI -> RGB + Height
% load '/media/thanu/raghavshdd1/2017-06-30_DJI/heightHealthy.mat'
% load '/media/thanu/raghavshdd1/2017-06-30_DJI/heightUnhealthy.mat'
% healthyName = '/media/thanu/raghavshdd1/2017-06-30_DJI/healthy_green.ply';
% unhealthyName = '/media/thanu/raghavshdd1/2017-06-30_DJI/unhealthy_plants.ply';
%
% healthyPC = plyread(healthyName);
% unhealthyPC = plyread(unhealthyName);
% healthy = healthyPC.Color;
% unhealthy = unhealthyPC.Color;

% TRAINING 03: DJI -> RGB + Height; with soil as category
load([hddLoc 'thanujan/Datasets/2017-06-30/DJI/heightHealthy.mat'])
load([hddLoc 'thanujan/Datasets/2017-06-30/DJI/heightUnhealthy.mat'])
healthyName = [hddLoc 'thanujan/Datasets/2017-06-30/DJI/healthy_green.ply'];
unhealthyName = [hddLoc 'thanujan/Datasets/2017-06-30/DJI/unhealthy_plants.ply'];
soilName = [hddLoc 'thanujan/Datasets/2017-05-18/DJI/soil_cloud.ply'];

healthyPC = plyread(healthyName);
unhealthyPC = plyread(unhealthyName);
soilPC = plyread(soilName);
healthy = healthyPC.Color;
unhealthy = unhealthyPC.Color;
soil = soilPC.Color;
heightSoil = soilPC.Location(:,3);

clear healthyName unhealthyName healthyPC unhealthyPC soilName soilPC

%% Preparing the Data
% Data for classification problems are set up for a neural network by
% organizing the data into two matrices, the input matrix X and the target
% matrix T.
%
% Each ith column of the input matrix will have 5 elements:
% [Height; NIR; REG; GRE; RED]
%
% Each corresponding column of the target matrix will have two elements.
% Unhealthy = [1;0] & healthy = [0;1]

% [x,t] = plant_dataset;

% TRAINING 01:
% x = [double(healthy);double(unhealthy)]';

% TRAINING 02:
% x = [double(healthy),heightHealthy;double(unhealthy),heightUnhealthy]';
%
% t = [repmat([0,1],size(healthy,1),1);repmat([1,0],size(unhealthy,1),1)]';

% TRAINING 03:
x = [double(healthy),heightHealthy;double(unhealthy),heightUnhealthy; double(soil),heightSoil]';

% Soil = [1;0;0]; unhealthy = [0;1;0] & healthy = [0;0;1]
t = [repmat([0,0,1],size(healthy,1),1);repmat([0,1,0],size(unhealthy,1),1);repmat([1,0,0],size(soil,1),1)]';

size(x)
size(t)

%% Building the Neural Network Classifier
% Since the neural network starts with random initial weights, the results
% will differ slightly every time it is run. The random seed
% is set to avoid this randomness.

setdemorandstream(491218382)

%% Specify number of hidden layers
% Two-layer (i.e. one-hidden-layer) feed forward neural networks can learn
% any input-output relationship given enough neurons in the hidden layer.
% Layers which are not output layers are called hidden layers.
%
% We will try a single hidden layer of 10 neurons for this example. In
% general, more difficult problems require more neurons, and perhaps more
% layers.  Simpler problems require fewer neurons.
%
% The input and output have sizes of 0 because the network has not yet
% been configured to match our input and target data.  This will happen
% when the network is trained.

net = patternnet(10);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Specify function to be used in hidden layer
net.layers{1}.transferFcn = 'purelin';

view(net)

%% Start training
% Now the network is ready to be trained. The samples are automatically
% divided into training, validation and test sets. The training set is
% used to teach the network. Training continues as long as the network
% continues improving on the validation set. The test set provides a
% completely independent measure of network accuracy.

[net,tr] = train(net,x,t); %,'useGPU','yes');
nntraintool

%% Plot performance
% To see how the network's performance improved during training, either
% click the "Performance" button in the training tool, or call PLOTPERFORM.
%
% Performance is measured in terms of mean squared error, and shown in
% log scale.  It rapidly decreased as the network was trained.
%
% Performance is shown for each of the training, validation and test sets.
% The version of the network that did best on the validation set is
% was after training.

plotperform(tr)

%% Testing the Classifier
% The trained neural network can now be tested with the testing samples
% This will give us a sense of how well the network will do when applied
% to data from the real world.
%
% The network outputs will be in the range 0 to 1, so we can use *vec2ind*
% function to get the class indices as the position of the highest element
% in each output vector.

testX = x(:,tr.testInd);
testT = t(:,tr.testInd);

testY = net(testX);
testIndices = vec2ind(testY);

%% Confusion Plot
% One measure of how well the neural network has fit the data is the
% confusion plot.  Here the confusion matrix is plotted across all samples.
%
% The confusion matrix shows the percentages of correct and incorrect
% classifications.  Correct classifications are the green squares on the
% matrices diagonal.  Incorrect classifications form the red squares.
%
% If the network has learned to classify properly, the percentages in the
% red squares should be very small, indicating few misclassifications.
%
% If this is not the case then further training, or training a network
% with more hidden neurons, would be advisable.

plotconfusion(testT,testY)

%% Overall percentages of correct and incorrect classification
[c,cm] = confusion(testT,testY)

fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

%% Receiver operating characteristic plot
% Another measure of how well the neural network has fit data is the
% receiver operating characteristic plot.  This shows how the false
% positive and true positive rates relate as the thresholding of outputs
% is varied from 0 to 1.
%
% The farther left and up the line is, the fewer false positives need to
% be accepted in order to get a high true positive rate.  The best
% classifiers will have a line going from the bottom left corner, to the
% top left corner, to the top right corner, or close to that.

plotroc(testT,testY)
