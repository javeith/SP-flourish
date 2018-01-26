%   augmentedImageSource Generate batches of augmented image data
%
%   source = augmentedImageSource(outputSize,imds) returns an
%   augmentedImageSource. outputSize is a two element vector which
%   specifies the output image size in the form [outputWidth,
%   outputHeight]. imds is an imageDatastore that contains both examples
%   and class labels.
%
%   source = augmentedImageSource(outputSize,X,Y) returns an
%   augmentedImageSource given matrices X and Y that define examples and 
%   corresponding responses.
%
%   source = augmentedImageSource(outputSize,tbl) returns a
%   an augmentedImageSource given a tbl which contains predictors in the 
%   first column as either absolute or relative image paths or images. 
%   Responses must be in the second column as categorical labels for the 
%   images. In a regression problem, responses must be in the second column 
%   as either vectors or cell arrays containing 3-D arrays or in multiple 
%   columns as scalars.
%
%   source = augmentedImageSource(outputSize,tbl, responseName,___)
%   returns an augmentedImageSource which yields predictors and responses. 
%   tbl is a MATLAB table. responseName is a character vector specifying
%   the name of the variable in tbl that contains the responses.
%
%   source = augmentedImageSource(outputSize,tbl,responseNames,___) returns
%   an augmentedImageSource for use in multi-output regression problems. 
%   tbl is a MATLAB table. responseNames is a cell array of character 
%   vectors specifying the names of the variables in tbl that contain the
%   responses.
%
%   source = augmentedImageSource(___,Name,Value) returns an
%   augmentedImageSource using Name/Value pairs to configure
%   image-preprocessing options.
%
%   Parameters include:
%
%   'BackgroundExecution'   Accelerate image augmentation by asyncronously
%                           reading, augmenting, and queueing augmented
%                           images for use in training. Requires Parallel
%                           Computing Toolbox.
%
%                           Default: false
%
%   'ColorPreprocessing'    A scalar string or character vector specifying
%                           color channel pre-processing. This option can
%                           be used when you have a training set that
%                           contains both color and grayscale image data
%                           and you need data created by the datasource to
%                           be strictly color or grayscale. Options are:
%                           'gray2rgb','rgb2gray','none'. For example, if
%                           you need to train a network that expects color
%                           images but some of the images in your training
%                           set are grayscale, then specifying the option
%                           'gray2rgb' will replicate the color channels of
%                           the grayscale images in the input image set to
%                           create MxNx3 output images.
%
%                           Default: 'none'
%
%   'DataAugmentation'      A scalar imageDataAugmenter object, string, or
%                           character array that specifies
%                           the kinds of image data augmentation that will
%                           be applied to generated images.
%
%                           Default: 'none'
%
%   'OutputSizeMode'        A scalar string or character vector specifying the
%                           technique used to adjust image sizes to the
%                           specified 'OutputSize'. Options are: 'resize',
%                           'centercrop', 'randcrop'.
%
%                           Default: 'resize'
%
%   Example 1
%   ---------
%   Train a convolutional neural network on some synthetic images of
%   handwritten digits. Apply random rotations during training to add
%   rotation invariance to trained network.
%
%   [XTrain, YTrain] = digitTrain4DArrayData;
%
%   imageSize = [28 28 1];
%
%   layers = [ ...
%       imageInputLayer(imageSize, 'Normalization', 'none');
%       convolution2dLayer(5,20);
%       reluLayer();
%       maxPooling2dLayer(2,'Stride',2);
%       fullyConnectedLayer(10);
%       softmaxLayer();
%       classificationLayer()];
%
%   opts = trainingOptions('sgdm');
%
%   imageAugmenter = imageDataAugmenter('RandRotation',[-10 10]);
%
%   source = augmentedImageSource(imageSize,XTrain,YTrain,'DataAugmentation',imageAugmenter);
%
%   net = trainNetwork(source, layers, opts);
%
% See also imageDataAugmenter, imageInputLayer, trainNetwork

% Copyright 2017 The MathWorks, Inc.

classdef augmentedImageSource < nnet.internal.cnn.MiniBatchDatasource &...
        nnet.internal.cnn.BackgroundDispatchableDatasource &...
        nnet.internal.cnn.DistributableAugmentedImageSource
    
    properties (Hidden, Dependent)
        MiniBatchSize
        NumberOfObservations
    end
    
    properties (Access = ?nnet.internal.cnn.DistributableMiniBatchDatasource)
        DatasourceInternal
    end
    
    properties (Access = private)
        ImageAugmenter
    end
    
    properties (SetAccess = private)
        
        %DataAugmentation - Augmentation applied to input images
        %
        %    DataAugmentation is a scalar imageDataAugmenter object or a
        %    character vector or string. When DataAugmentation is 'none' 
        %    no augmentation is applied to input images.
        DataAugmentation
        
        %ColorPreprocessing - Pre-processing of input image color channels
        %
        %    ColorPreprocessing is a character vector or string specifying
        %    pre-proprocessing operations performed on color channels of
        %    input images. This property is used to ensure that all output
        %    images from the datasource have the number of color channels
        %    required by inputImageLayer. Valid values are
        %    'gray2rgb','rgb2gray', and 'none'. If an input images already
        %    has the desired number of color channels, no operation is
        %    performed. For example, if 'gray2rgb' is specified and an
        %    input image already has 3 channels, no operation is performed.
        ColorPreprocessing
        
        %OutputSize - Size of output images
        %
        %    OutputSize is a two element numeric vector of the form
        %    [numRows, numColumns] that specifies the size of output images
        %    returned by augmentedImageSource.
        OutputSize
        
        %OutputSizeMode - Method used to resize output images.
        %
        %    OutputSizeMode is a character vector or string specifying the
        %    method used to resize output images to the requested
        %    OutputSize. Valid values are 'centercrop', 'randcrop', and 
        %   'resize' (default).
        OutputSizeMode
    end
    
    properties (Access = private)
        OutputRowsColsChannels % The expected output image size [numRows, numCols, numChannels].
    end
    
    methods
        function self = augmentedImageSource(varargin)
            inputs = self.parseInputs(varargin{:});
            self.ImageAugmenter = inputs.DataAugmentation;
            
            self.determineExpectedOutputSize();
        end
        
        function set.MiniBatchSize(self,batchSize)
            self.DatasourceInternal.MiniBatchSize = batchSize;
        end
        
        function batchSize = get.MiniBatchSize(self)
            batchSize = self.DatasourceInternal.MiniBatchSize;
        end
        
        function numObs = get.NumberOfObservations(self)
            numObs = self.DatasourceInternal.NumberOfObservations;
        end
        
    end
    
    methods (Hidden)
        function reset(self)
            self.DatasourceInternal.reset();
        end
        
        function [X,Y,indices] = getObservations(self,indices)
            [Xin,Y] = self.DatasourceInternal.getObservations(indices);
            X = self.applyAugmentationPipelineToBatch(Xin);
        end
        
        function [X,Y] = nextBatch(self)
            [Xin,Y] = self.DatasourceInternal.nextBatch();
            X = self.applyAugmentationPipelineToBatch(Xin);
        end
        
        function shuffle(self)
            self.DatasourceInternal.shuffle();
        end
        
        function reorder(self,indices)
            self.DatasourceInternal.reorder(indices);
        end
        
    end
    
    methods (Access = private)
        
        function determineExpectedOutputSize(self)
            
            % If a user specifies a ColorPreprocessing option, we know the
            % number of channels to expect in each mini-batch. If they
            % don't specify a ColorPreprocessing option, we need to look at
            % an example from the underlying Datasource and assume all
            % images will have a consistent number of channels when forming
            % mini-batches.
            if strcmp(self.ColorPreprocessing,'rgb2gray')
                self.OutputRowsColsChannels = [self.OutputSize,1];
            elseif strcmp(self.ColorPreprocessing,'gray2rgb')
                self.OutputRowsColsChannels = [self.OutputSize,3];
            elseif strcmp(self.ColorPreprocessing,'none')
                origMiniBatchSize = self.MiniBatchSize;
                self.DatasourceInternal.MiniBatchSize = 1;
                X = self.DatasourceInternal.nextBatch();
                self.DatasourceInternal.MiniBatchSize = origMiniBatchSize;
                self.DatasourceInternal.reset();
                exampleNumChannels = size(X,3);
                self.OutputRowsColsChannels = [self.OutputSize,exampleNumChannels];
            else
                assert(false,'Unexpected ColorPreprocessing option.');
            end
            
        end
        
        function Xout = applyAugmentationPipelineToBatch(self,X)
            if iscell(X)
                Xout = cellfun(@(c) self.applyAugmentationPipeline(c),X,'UniformOutput',false);
            else
                batchSize = size(X,4);
                Xout = zeros(self.OutputRowsColsChannels,'like',X);
                for obs = 1:batchSize
                    temp = self.preprocessColor(X(:,:,:,obs));
                    temp = self.augmentData(temp);
                    Xout(:,:,:,obs) = self.resizeData(temp);
                end
            end
        end
        
        function Xout = applyAugmentationPipeline(self,X)
            if isequal(self.ColorPreprocessing,'none') && (size(X,3) ~= self.OutputRowsColsChannels(3))
               error(message('nnet_cnn:augmentedImageSource:mismatchedNumberOfChannels','''ColorPreprocessing'''));
            end
            temp = self.preprocessColor(X);
            temp = self.augmentData(temp);
            Xout = self.resizeData(temp);
        end
        
        function miniBatchData = augmentData(self,miniBatchData)
            if ~strcmp(self.DataAugmentation,'none')
                miniBatchData = self.ImageAugmenter.augment(miniBatchData);
            end
        end
        
        function Xout = resizeData(self,X)
            
            inputSize = size(X);
            if isequal(inputSize(1:2),self.OutputSize)
                Xout = X; % no-op if X is already desired Outputsize
                return
            end
            
            if strcmp(self.OutputSizeMode,'resize')
                Xout = augmentedImageSource.resizeImage(X,self.OutputSize);
            elseif strcmp(self.OutputSizeMode,'centercrop')
                Xout = augmentedImageSource.centerCrop(X,self.OutputSize);
            elseif strcmp(self.OutputSizeMode,'randcrop')
                Xout = augmentedImageSource.randCrop(X,self.OutputSize);
            end
        end
        
        function Xout = preprocessColor(self,X)
            
            if strcmp(self.ColorPreprocessing,'rgb2gray')
                Xout = convertRGBToGrayscale(X);
            elseif strcmp(self.ColorPreprocessing,'gray2rgb')
                Xout = convertGrayscaleToRGB(X);
            elseif strcmp(self.ColorPreprocessing,'none')
                Xout = X;
            end
        end
    end
    
    methods (Access = 'private')
        
        function inputStruct = parseInputs(self,varargin)
            
            narginchk(2,inf) % Use input parser to validate upper end of range.
            
            p = inputParser();
              
            p.addRequired('outputSize',@outputSizeValidator);
            p.addRequired('X');
            p.addOptional('Y',[]);
            p.addParameter('DataAugmentation','none',@augmentationValidator);
            
            colorPreprocessing = 'none';
            p.addParameter('ColorPreprocessing','none',@colorPreprocessingValidator);
            
            
            outputSizeMode = 'resize';
            p.addParameter('OutputSizeMode','resize',@outputSizeModeValidator);
            
            backgroundExecutionValidator = @(TF) validateattributes(TF,...
                {'numeric','logical'},{'scalar','real'},mfilename,'BackgroundExecution');
            p.addParameter('BackgroundExecution',false,backgroundExecutionValidator);
            
            responseNames = [];
            if istable(varargin{2})
                tbl = varargin{2};
                if (length(varargin) > 2) && (ischar(varargin{3}) || isstring(varargin{3}) || iscellstr(varargin{3}))
                    if checkValidResponseNames(varargin{3},tbl)
                        responseNames = varargin{3};
                        varargin(3) = [];
                    end
                end
            end
            
            p.parse(varargin{:});
            inputStruct = p.Results;
            
            self.DataAugmentation = inputStruct.DataAugmentation;
            self.OutputSize = inputStruct.outputSize(1:2);
            self.OutputSizeMode = outputSizeMode;
            self.ColorPreprocessing = colorPreprocessing;
            self.UseParallel = inputStruct.BackgroundExecution;
            
            % Check if Y was specified for table or imageDatastore inputs.
            propertiesWithDefaultValues = string(p.UsingDefaults);
            if (isa(inputStruct.X,'matlab.io.datastore.ImageDatastore') || isa(inputStruct.X,'table')) && ~any(propertiesWithDefaultValues == "Y")
                error(message('nnet_cnn:augmentedImageSource:invalidYSpecification',class(inputStruct.X)));
            end
            
            if ~isempty(responseNames)
                inputStruct.X = selectResponsesFromTable(inputStruct.X,responseNames);
                inputStruct.Y = responseNames;
            end
            
            % Validate numeric inputs
            if isnumeric(inputStruct.X)
                validateattributes(inputStruct.X,{'single','double','logical','uint8','int8','uint16','int16','uint32','int32'},...
                    {'nonsparse','real'},mfilename,'X');
                
                validateattributes(inputStruct.Y,{'single','double','logical','uint8','int8','uint16','int16','uint32','int32','categorical'},...
                    {'nonsparse','nonempty'},mfilename,'Y');
            end
                            
            try
                self.DatasourceInternal = nnet.internal.cnn.MiniBatchDatasourceFactory.createMiniBatchDatasource(inputStruct.X,inputStruct.Y);
            catch ME
                throwAsCaller(ME);
            end
            
            function TF = colorPreprocessingValidator(sIn)
                colorPreprocessing = validatestring(sIn,{'none','rgb2gray','gray2rgb'},...
                    mfilename,'ColorPreprocessing');
                
                TF = true;
            end
            
            function TF = outputSizeModeValidator(sIn)
                outputSizeMode = validatestring(sIn,...
                    {'resize','centercrop','randcrop'},mfilename,'OutputSizeMode');
                
                TF = true;
            end
            
            function TF = outputSizeValidator(sizeIn)
               
                validateattributes(sizeIn,...
                {'numeric'},{'vector','integer','finite','nonsparse','real','positive'},mfilename,'OutputSize');
            
                if (numel(sizeIn) ~= 2) && (numel(sizeIn) ~=3)
                   error(message('nnet_cnn:augmentedImageSource:invalidOutputSize')); 
                end
                
                TF = true;
                
            end
            
        end
        
    end
    
    methods(Static, Hidden = true)
        function self = loadobj(S)
            self = augmentedImageSource(S.OutputSize,S.DatasourceInternal,...
                'BackgroundExecution',S.BackgroundExecution,...
                'ColorPreprocessing',S.ColorPreprocessing,...
                'DataAugmentation',S.DataAugmentation,...
                'OutputSizeMode',S.OutputSizeMode);            
        end
    end
    
    methods (Hidden)
        function S = saveobj(self)
            S = struct('BackgroundExecution',self.UseParallel,...
                'ColorPreprocessing',self.ColorPreprocessing,...
                'DataAugmentation',self.DataAugmentation,...
                'OutputSize',self.OutputSize,...
                'OutputSizeMode',self.OutputSizeMode,...
                'DatasourceInternal',self.DatasourceInternal);
        end
        
    end
    
    methods (Hidden, Static)
        
        
        function imOut = resizeImage(im,outputSize)
            
            ippResizeSupportedWithCast = isa(im,'int8') || isa(im,'uint16') || isa(im,'int16');
            ippResizeSupportedForType = isa(im,'uint8') || isa(im,'single');
            ippResizeSupported = ippResizeSupportedWithCast || ippResizeSupportedForType;
            
            if ippResizeSupportedWithCast
                im = single(im);
            end
            
            if ippResizeSupported
                imOut = nnet.internal.cnnhost.resizeImage2D(im,outputSize,'linear',true);
            else
                imOut = imresize(im,'OutputSize',outputSize,'method','bilinear');
            end
            
        end
        
        function im = centerCrop(im,outputSize)
            
            sizeInput = size(im);
            if any(sizeInput(1:2) < outputSize)
               error(message('nnet_cnn:augmentedImageSource:invalidCropOutputSize','''OutputSizeMode''',mfilename, '''centercrop''','''OutputSize''')); 
            end
            
            x = (size(im,2) - outputSize(2)) / 2;
            y = (size(im,1) - outputSize(1)) / 2;
                        
            im = augmentedImageSource.crop(im,...
                [x y, outputSize(2), outputSize(1)]);
        end
        
        function rect = randCropRect(im,outputSize)
            % Pick random coordinates within the image bounds
            % for the top-left corner of the cropping rectangle.
            range_x = size(im,2) - outputSize(2);
            range_y = size(im,1) - outputSize(1);
            
            x = range_x * rand;
            y = range_y * rand;
            rect = [x y outputSize(2), outputSize(1)];
        end
                     
        function im = randCrop(im,outputSize)
            sizeInput = size(im);
            if any(sizeInput(1:2) < outputSize)
                error(message('nnet_cnn:augmentedImageSource:invalidCropOutputSize','''OutputSizeMode''',mfilename, '''randcrop''','''OutputSize'''));
            end
            rect = augmentedImageSource.randCropRect(im,outputSize);
            im = augmentedImageSource.crop(im,rect);
        end
        
        function B = crop(A,rect)
            % rect is [x y width height] in floating point.
            % Convert from (x,y) real coordinates to [m,n] indices.
            rect = floor(rect);
            
            m1 = rect(2) + 1;
            m2 = rect(2) + rect(4);
            
            n1 = rect(1) + 1;
            n2 = rect(1) + rect(3);
                        
            m1 = min(size(A,1),max(1,m1));
            m2 = min(size(A,1),max(1,m2));
            n1 = min(size(A,2),max(1,n1));
            n2 = min(size(A,2),max(1,n2));
            
            B = A(m1:m2, n1:n2, :, :);
        end
    end
end

function TF = checkValidResponseNames(responseNames, tbl)
% iAssertValidResponseNames   Assert that the response names are variables
% of the table and they do not refer to the first column.
variableNames = tbl.Properties.VariableNames;
refersToFirstColumn = ismember( variableNames(1), responseNames );
responseNamesAreAllVariables = all( ismember(responseNames,variableNames) );
TF = ~(refersToFirstColumn || ~responseNamesAreAllVariables);
end

function resTbl = selectResponsesFromTable(tbl, responseNames)
% iSelectResponsesFromTable   Return a new table with only the first column
% (predictors) and the variables specified in responseNames.
variableNames = tbl.Properties.VariableNames;
varTF = ismember(variableNames, responseNames);
% Make sure to select predictors (first column) as well
varTF(1) = 1;
resTbl = tbl(:,varTF);
end

function TF = augmentationValidator(valIn)

if ischar(valIn) || isstring(valIn)
    TF = string('none').contains(lower(valIn)); %#ok<STRQUOT>
elseif isa(valIn,'imageDataAugmenter') && isscalar(valIn)
    TF = true;
else
    TF = false;
end

end

function im = convertRGBToGrayscale(im)
if ndims(im) == 3
    im = rgb2gray(im);
end
end

function im = convertGrayscaleToRGB(im)
if size(im,3) == 1
    im = repmat(im,[1 1 3]);
end
end

