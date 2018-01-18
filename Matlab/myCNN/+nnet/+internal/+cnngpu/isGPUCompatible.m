function tf = isGPUCompatible(errorForPascal)
% isGPUCompatible   Check if the currently selected GPU is compatible
%   tf = isGPUCompatible() will return true if the currently
%   selected GPU device can be used with the Convolutional Neural Network
%   feature, which requires an NVIDIA GPU with compute capability 3.0

%   Copyright 2016 The MathWorks, Inc.

if(canUsePCT() && parallel.gpu.GPUDevice.isAvailable())
    gpuInfo = gpuDevice();
    tf = iComputeCapabilityIsGreaterThanOrEqualToThree(gpuInfo) && ...
        iComputeCapabilityIsLessThanSix(gpuInfo);
    % Once the NVIDIA bug with Pascal cards is fixed, we can remove the
    % next block of code (and the above check for compute capability < 6).
    if(~iComputeCapabilityIsLessThanSix(gpuInfo))
        if(errorForPascal)
            error(message('nnet_cnn:internal:cnngpu:PascalCardsNotSupported'));
        else
            warning(message('nnet_cnn:internal:cnngpu:PascalCardsNotSupportedSwitchingToCPU'));
            warning('off', 'nnet_cnn:internal:cnngpu:PascalCardsNotSupportedSwitchingToCPU');
        end
    end
else
    tf = false;
end
end

function tf = iComputeCapabilityIsGreaterThanOrEqualToThree(gpuInfo)
tf = str2double(gpuInfo.ComputeCapability) >= 3.0;
end

function tf = iComputeCapabilityIsLessThanSix(gpuInfo)
tf = str2double(gpuInfo.ComputeCapability) < 6.0;
end

function ok = canUsePCT()
%canUsePCT  Check that Parallel Computing Toolbox is installed and licensed
 
% Checking for installation is expensive, so only do it once
persistent pctInstalled;
if isempty(pctInstalled)
    pctInstalled = exist('gpuArray', 'file') == 2;
end
 
% Check the license every time as it may have changed
pctLicensed = license('test', 'Distrib_Computing_Toolbox');
 
% Now see if everything is OK with the hardware
ok = pctInstalled && pctLicensed;

end