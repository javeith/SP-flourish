%% Filter images with geolocation for Pix4D

clc, clear

file = fopen('data/2017-04-07/imagesColor.txt', 'r');

C = textscan(file, '%s','delimiter', '\n');
s = cellfun('length',C)

for i = 1:s
    copyfile(['data/2017-04-07/ximea/' C{1}{i}],'data/2017-04-07/ximea_Pix4D')
end