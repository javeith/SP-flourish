function [ imagesDeleted ] = checkNIR25VIS16Pairs( NIRLoc, VISLoc )
%% Check NIR25 <--> VIS16 image pairs

%% Read files
% NIR
NIRFiles = dir(NIRLoc);
[~,ndx] = natsortfiles({NIRFiles.name});
NIRFiles = NIRFiles(ndx);
NoI_NIR = length(NIRFiles);

% VIS
VISFiles = dir(VISLoc);
[~,ndx] = natsortfiles({VISFiles.name});
VISFiles = VISFiles(ndx);
NoI_VIS = length(VISFiles);

%% Check pairs
for iNIR = 1:NoI_NIR
    NIR_ID(iNIR) = str2double(extractBetween(NIRFiles(iNIR).name,'_','_'));
end

for iVIS = 1:NoI_VIS
    VIS_ID(iVIS) = str2double(extractBetween(VISFiles(iVIS).name,'_','_'));
end

[~,noPairsNIR] = setdiff(NIR_ID,VIS_ID);
[~,noPairsVIS] = setdiff(VIS_ID,NIR_ID);

imagesDeleted = size(noPairsNIR,1) + size(noPairsVIS,1);

for iNIR = 1:size(noPairsNIR,1)
    fileName = [NIRFiles(noPairsNIR(iNIR)).folder '/' NIRFiles(noPairsNIR(iNIR)).name];
    delete(fileName)
end

for iVIS = 1:size(noPairsVIS,1)
    fileName = [VISFiles(noPairsVIS(iVIS)).folder '/' VISFiles(noPairsVIS(iVIS)).name];
    delete(fileName)
end

end