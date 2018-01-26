% MiniBatchDatasourceFactory   Factory for making MiniBatchDatasources
%
%   mbds = MiniBatchDatasourceFactoryInstance.createMiniBatchDatasource(data)
%   data: the data to be dispatched.
%       According to their type the appropriate MiniBatchDatasource will be used.
%       Supported types: 4-D double, imageDatastore, table.

%   Copyright 2017 The MathWorks, Inc.

classdef MiniBatchDatasourceFactory
     
    methods (Static)
        function mbds = createMiniBatchDatasource( inputs, response, initMiniBatchSize)
            % createMiniBatchDatasource   Create MiniBatchDatasource
            %
            % Syntax:
            %     createMiniBatchDatasource(inputs, response)
            
            if nargin < 3
                initMiniBatchSize = 128;
            end
            
            if iIsRealNumeric4DHostArray(inputs)
                 mbds = nnet.internal.cnn.FourDArrayMiniBatchDatasource(inputs, response, initMiniBatchSize);
            elseif isa(inputs, 'matlab.io.datastore.ImageDatastore')
                mbds  = nnet.internal.cnn.ImageDatastoreMiniBatchDatasource(inputs, initMiniBatchSize);
            elseif istable(inputs)
                if iIsAnInMemoryTable(inputs)
                    mbds = nnet.internal.cnn.InMemoryTableMiniBatchDatasource(inputs, initMiniBatchSize);
                else
                    mbds = nnet.internal.cnn.FilePathTableMiniBatchDatasource(inputs, initMiniBatchSize);
                end
            elseif isa(inputs,'nnet.internal.cnn.MiniBatchDatasource')
                % If passed MiniBatchDatasource, use it. Used for
                % layered compositions of MiniBatchDatasource.
                mbds = inputs;
            else
               error( message( 'nnet_cnn:internal:cnn:MiniBatchDatasourceFactory:InvalidData' ) );
            end
           
        end
        
    end
    
end

function tf = iIsAnInMemoryTable( x )
firstCell = x{1,1};
tf = isnumeric( firstCell{:} );
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
tf = ( iIsGrayscale( x ) || iIsColour( x ) ) && ...
    iIs4DArray( x );
end

function tf = iIsGrayscale(x)
tf = size(x,3)==1;
end

function tf = iIsColour(x)
tf = size(x,3)==3;
end

function tf = iIs4DArray(x)
tf = ndims(x) <= 4;
end

function tf = iIsGPUArray( x )
tf = isa(x, 'gpuArray');
end