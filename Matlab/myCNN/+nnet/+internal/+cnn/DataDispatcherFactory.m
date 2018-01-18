classdef DataDispatcherFactory
    % DataDispatcherFactory   Factory for making data dispatchers
    %
    %   dataDispatcher = DataDispatcherFactoryInstance.createDataDispatcher(data, options)
    %   data: the data to be dispatched.
    %       According to their type the appropriate dispatcher will be used.
    %       Supported types: 4-D double, imagedatastore, ...
    %   options: input arguments for the data dispatcher (e.g. response vector,
    %   mini batch size)
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    methods (Static)
        function dispatcher = createDataDispatcher( inputs, response, miniBatchSize, endOfEpoch, precision )
            % createDataDispatcher   Create data dispatcher
            %
            % Syntax:
            %     createDataDispatcher( inputs, response, miniBatchSize, endOfEpoch, precision )
            
            if iIsRealNumeric4DHostArray(inputs)
                dispatcher = iCreate4dArrayDispatcher( inputs, response, miniBatchSize, endOfEpoch, precision );
            elseif isa(inputs, 'matlab.io.datastore.ImageDatastore')
                dispatcher  = iCreateImageDatastoreDispatcher( inputs, miniBatchSize, endOfEpoch, precision );
            elseif isa(inputs, 'nnet.internal.cnn.DataDispatcher')
                dispatcher = inputs;
                
                % Setup the dispatcher to factory specifications.
                dispatcher.EndOfEpoch    = endOfEpoch;
                dispatcher.Precision     = precision;
                dispatcher.MiniBatchSize = miniBatchSize;
            else
                error('nnet_cnn:internal:cnn:DataDispatcherFactory:InvalidData', 'Invalid input data type')
            end
        end
    end
end

function dispatcher = iCreate4dArrayDispatcher( inputs, response, miniBatchSize, endOfEpoch, precision )
dispatcher = nnet.internal.cnn.FourDArrayDispatcher( inputs, response, miniBatchSize, endOfEpoch, precision );
end

function dispatcher = iCreateImageDatastoreDispatcher( inputs, miniBatchSize, endOfEpoch, precision )
dispatcher = nnet.internal.cnn.ImageDatastoreDispatcher( inputs, miniBatchSize, endOfEpoch, precision );
end

function tf = iIsRealNumeric4DHostArray( x )
tf = iIsRealNumericData( x ) && iIsValidImageArray( x ) && ~iIsGPUArray( x );
end

function tf = iIsRealNumericData(x)
tf = isreal(x) && isnumeric(x) && ~issparse(x);
end

function tf = iIsValidImageArray(x)
% iIsValidImageArray   Return true if x is an array of
% one or multiple (colour or grayscale) images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Instead of Grayscale and Colour, accept any number of image channels. 
% Check for the channels to be bigger than 0.
%tf = ( iIsGrayscale( x ) || iIsColour( x ) ) && ...
%    iIs4DArray( x );
tf = iChannelSizeIsBiggerThanZero( x ) && iIs4DArray( x );
end

function tf = iChannelSizeIsBiggerThanZero(x)
tf = size(x,3) > 0;
end
 
% function tf = iIsGrayscale(x)
% tf = size(x,3)==1;
% end
% 
% function tf = iIsColour(x)
% tf = size(x,3)==3;
% end

function tf = iIs4DArray(x)
sz = size( x );
tf = numel( sz ) <= 4;
end

function tf = iIsGPUArray( x )
tf = isa(x, 'gpuArray');
end