
%    Created on: July 17, 2017
%    Author: Thanujan Mohanadasan
%    Institute: ETH Zurich, Autonomous Systems Lab

%% Healthy Plant Classification
clc, clear
close all

%% raghavshdd1 location
hddLoc = '/media/thanu/raghavshdd1/';

%% Inputs:
% TRAINING 01:
for iBand = 1:25
    corn1_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Corn1/NIR25/crop1_1_band'  num2str(iBand) '.ply']);
    size1 = size(corn1_pc.Color,1);
    corn(iBand,1:size1) = corn1_pc.Color(:,1);
    clear corn1_pc;
    
    corn2_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Corn2/NIR25/crop1_2_band'  num2str(iBand) '.ply']);
    corn(iBand,(size1+1):(size1+size(corn2_pc.Color,1))) = corn2_pc.Color(:,1);
    clear corn2_pc size1;
    
    sugarBeet_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Sugarbeet/NIR25/crop2_1_band'  num2str(iBand) '.ply']);
    sugarBeet(iBand,:) = sugarBeet_pc.Color(:,1);
    clear sugarBeet_pc;
    
    winterWheat1_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Winterwheat1/NIR25/crop3_1_band'  num2str(iBand) '.ply']);
    size2 = size(winterWheat1_pc.Color,1);
    winterWheat(iBand,1:size2) = winterWheat1_pc.Color(:,1);
    clear winterWheat1_pc;
    
    winterWheat2_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Winterwheat2/NIR25/winterwheat2_band'  num2str(iBand) '.ply']);
    winterWheat(iBand,(size2+1):(size2+size(winterWheat2_pc.Color,1))) = winterWheat2_pc.Color(:,1);
    clear winterWheat2_pc size2;
    
    buckWheat_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Buckwheat/NIR25/buckwheat_band'  num2str(iBand) '.ply']);
    buckWheat(iBand,:) = buckWheat_pc.Color(:,1);
    clear buckWheat_pc;
    
    grass_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Grass/NIR25/grass_band'  num2str(iBand) '.ply']);
    grass(iBand,:) = grass_pc.Color(:,1);
    clear grass_pc;
    
    road_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/road/NIR25/road_band'  num2str(iBand) '.ply']);
    road(iBand,:) = road_pc.Color(:,1);
    clear road_pc;
    
    soil_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/soil/NIR25/soil_band'  num2str(iBand) '.ply']);
    soil(iBand,:) = soil_pc.Color(:,1);
    clear soil_pc;
    
    soy_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Soybean/NIR25/soybean_band'  num2str(iBand) '.ply']);
    soyBean(iBand,:) = soy_pc.Color(:,1);
    clear soy_pc;
    
end

for iBand = 1:16
    corn1_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Corn1/VIS16/band'  num2str(iBand) '.ply']);
    size1 = size(corn1_pc.Color,1);
    corn(iBand+25,1:size1) = corn1_pc.Color(:,1);
    clear corn1_pc;
    
    corn2_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Corn2/VIS16/band'  num2str(iBand) '.ply']);
    corn(iBand+25,(size1+1):(size1+size(corn2_pc.Color,1))) = corn2_pc.Color(:,1);
    clear corn2_pc size1;
    
    sugarBeet_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Sugarbeet/VIS16/band'  num2str(iBand) '.ply']);
    sugarBeet(iBand+25,:) = sugarBeet_pc.Color(:,1);
    clear sugarBeet_pc;
    
    winterWheat1_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Winterwheat1/VIS16/band'  num2str(iBand) '.ply']);
    size2 = size(winterWheat1_pc.Color,1);
    winterWheat(iBand+25,1:size2) = winterWheat1_pc.Color(:,1);
    clear winterWheat1_pc;
    
    winterWheat2_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Winterwheat2/VIS16/winterwheat2_band'  num2str(iBand) '.ply']);
    winterWheat(iBand+25,(size2+1):(size2+size(winterWheat2_pc.Color,1))) = winterWheat2_pc.Color(:,1);
    clear winterWheat2_pc size2;
    
    buckWheat_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Buckwheat/VIS16/buckwheat_band'  num2str(iBand) '.ply']);
    buckWheat(iBand+25,:) = buckWheat_pc.Color(:,1);
    clear buckWheat_pc;
    
    grass_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Grass/VIS16/grass_band'  num2str(iBand) '.ply']);
    grass(iBand+25,:) = grass_pc.Color(:,1);
    clear grass_pc;
    
    road_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/road/VIS16/road_band'  num2str(iBand) '.ply']);
    road(iBand+25,:) = road_pc.Color(:,1);
    clear road_pc;
    
    soil_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/soil/VIS16/soil_band'  num2str(iBand) '.ply']);
    soil(iBand+25,:) = soil_pc.Color(:,1);
    clear soil_pc;
    
    soy_pc = plyread([hddLoc 'thanujan/Datasets/xClassifier/trainSet/Soybean/VIS16/soybean_band'  num2str(iBand) '.ply']);
    soyBean(iBand+25,:) = soy_pc.Color(:,1);
    clear soy_pc;
    
end

clear iBand

%% Preparing the Data
% Data for classification problems are set up for a neural network by
% organizing the data into two matrices, the input matrix X and the target
% matrix T.
%
% Each ith column of the input matrix will have 41 elements:
% [41 Bands]
%
% Each corresponding column of the target matrix will have two elements.
% Soil = [1;0;0;0;0], road = [0;1;0;0;0], crop1 = [0;0;1;0;0], crop2 = [0;0;0;1;0], crop3 = [0;0;0;0;1]

% TRAINING 01:
x = double([soil, road, buckWheat, corn, grass, soyBean, sugarBeet, winterWheat]);
t = [repmat([1,0,0,0,0,0,0,0],size(soil,2),1);repmat([0,1,0,0,0,0,0,0],size(road,2),1);repmat([0,0,1,0,0,0,0,0],size(buckWheat,2),1); ...
    repmat([0,0,0,1,0,0,0,0],size(corn,2),1); repmat([0,0,0,0,1,0,0,0],size(grass,2),1); repmat([0,0,0,0,0,1,0,0],size(soyBean,2),1); ...
    repmat([0,0,0,0,0,0,1,0],size(sugarBeet,2),1); repmat([0,0,0,0,0,0,0,1],size(winterWheat,2),1)]';

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

net = patternnet(20);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Specify function to be used in hidden layer
% net.layers{1}.transferFcn = 'purelin';

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
set(gca,'fontsize',18);

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
set(gca,'fontsize',18);

%% Overall percentages of correct and incorrect classification
[corn1_pc,cm] = confusion(testT,testY)

fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-corn1_pc));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*corn1_pc);

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
axisdata = get(gca,'userdata');
set(gca,'fontsize',18);
legend(axisdata.lines,'Soil', 'Road', 'Buckwheat', 'Corn', 'Grass', 'Soybean', 'Sugarbeet', 'Winter wheat')

%% Precision-Recall curve
classes = {'Soil', 'Road', 'Buckwheat', 'Corn', 'Grass', 'Soybean', 'Sugarbeet', 'Winter wheat'};
for i = 1:size(testT,1)
    prec_rec(testY(i,:),testT(i,:),'holdFigure', 1, 'plotROC', 0, 'plotPR', 1, 'numThresh', 1000, 'plotBaseline', 0);
end
set(gca,'fontsize',18);
legend(classes);
grid on
