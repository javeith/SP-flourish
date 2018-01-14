%% Healthy Plant Classification using Discriminant Analysis
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
% Each ith column of the input matrix will have 41 elements:
% [41 Bands]

% TRAINING 01:
x = double([soil, road, buckWheat, corn, grass, soyBean, sugarBeet, winterWheat])';
t = [repmat({'soil'},size(soil,2),1);repmat({'road'},size(road,2),1);repmat({'buckWheat'},size(buckWheat,2),1); ...
    repmat({'corn'},size(corn,2),1); repmat({'grass'},size(grass,2),1);repmat({'soyBean'},size(soyBean,2),1); ...
    repmat({'sugarBeet'},size(sugarBeet,2),1); repmat({'winterWheat'},size(winterWheat,2),1)];

%% Linear Classifier
MdlLinear = fitcdiscr(x,t); %,'Prior','Uniform');

MdlLinear.ClassNames([1 2])
K = MdlLinear.Coeffs(1,2).Const;
L = MdlLinear.Coeffs(1,2).Linear;

% Scatter plot
h1 = gscatter(x(1:1000:end,8),x(1:1000:end,14),t(1:1000:end));
hold on

f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h3 = ezplot(f,[0 255 0 255]);
h3.Color = 'k';
h3.LineWidth = 2;
xlabel('Band 8')
ylabel('Band 14')
title('{\bf Linear Classification}')

%% Quadratic Classifier
MdlQuadratic = fitcdiscr(x, t, 'DiscrimType', 'quadratic'); %,'Prior','Uniform');

%% Importance of attributes using ReliefF algorithm
% Find K (number of neighbors)
% for K = 500:10:700
%     [ranked,weights(:,K)] = relieff(x(1:1000:end,:),t(1:1000:end),K);
%
%     % Display information
%     if mod(K,10) <= 1
%         disp([num2str(K) ' out of ' num2str(1000)]);
%     end
% end
%
% xPlot = 500:10:700;
% figure(1)
% set(gca,'fontsize',18)
% hold on
% for iBand = 1:41
%     plot(xPlot, weights(iBand,xPlot), '-o', 'DisplayName', ['band ' num2str(iBand)], 'LineWidth', 2, 'MarkerSize', 5);
% end
% fig = get(gca,'Children');
% ax1 = gca;
% fig = flipud(fig);
% a = legend(fig(1:25));
% xlabel('Number of nearest neighbors K')
% ylabel('Weight of each band depending on K')
% title('Estimation of K')
% grid on
% ax2 = axes('Position',get(ax1,'Position'),...
%     'Visible','off','Color','none');
% set(gca,'fontsize',18)
% b = legend(ax2,fig(26:end));
% hold off

% Normalizing Input vector
denom = mean(x,2);
x = x./denom;
[ranked,realweights] = relieff(x(1:100:end,:),t(1:100:end),600);
NIR25 = [700   714   739   752   765   778   791   803   816   828   845   857   867   864   791   608   615   623   636   645   653   662   670   678   686];
VIS16 = [465    474    485    496    510    522    534    546    548    562    578    586    600    608    624    630];
[NIR25_sorted,ind] = sort(NIR25);
weights_sorted = realweights(ind);

figure(1)
set(gca,'fontsize',18)
hold on
bar([1:14,18:19],realweights(26:end),'r');
bar([15:17,20:41],weights_sorted);

xlabel('Wavelength [nm]');
ylabel('Predictor importance weight');
legend('VIS16','NIR25')
title('{\bf Importance of Bands for Linear Classification}')
wavelengths = sort([NIR25,VIS16]);
wavelengths = num2str(wavelengths');
set(gca, 'XTick', [1:41], 'XTickLabel', wavelengths, 'XTickLabelRotation', 90)
xticklabels(wavelengths)
grid on
hold off

%% SVM --> Works only for two classes
% SVMModel = fitcsvm([x(:,8),x(:,5)],t,'KernelFunction','rbf','Standardize',true);