classdef FourDArrayMiniBatchDatasource <...
         nnet.internal.cnn.MiniBatchDatasource &...
         nnet.internal.cnn.DistributableFourDArrayMiniBatchDatasource &...
         nnet.internal.cnn.BackgroundDispatchableDatasource
       
     % FourDArrayMiniBatchDatasource class to return 4D data one mini batch at a
     %   time from 4D numeric data
     %
     % Input data    - 4D data where the last dimension is the number of
     %               observations.
     % Output data   - 4D data where the last dimension is the number of
     %               observations in that mini batch. The type of the data
     %               in output will be the same as the one in input
     
     %   Copyright 2017 The MathWorks, Inc.
     
    properties (Dependent)
       MiniBatchSize 
    end
        
    properties
        NumberOfObservations
    end
    
    properties (Access = private)
        StartIndexOfCurrentMiniBatch 
        MiniBatchSizeInternal
    end
    
    properties (Access = ?nnet.internal.cnn.DistributableMiniBatchDatasource)
        Input
        Response
        OrderedIndices % Shuffled sequence of all indices into observations
    end
    
    methods
        
        function self = FourDArrayMiniBatchDatasource(X,Y,miniBatchSize)
            
            if ~isempty(X)
                iValidateNumericInputs(X,Y);
                self.Input = X;
                self.Response = Y;
                self.NumberOfObservations = size(X,4);
                self.StartIndexOfCurrentMiniBatch = 1;
                self.OrderedIndices = 1:self.NumberOfObservations;
                self.MiniBatchSize = miniBatchSize;
            else
                self = nnet.internal.cnn.FourDArrayMiniBatchDatasource.empty();
            end
        end
        
        function [X,Y] = getObservations(self,indices)
            % getObservations  Overload of method to retrieve specific
            % observations
            
            [X,Y] = self.readData(self.OrderedIndices(indices));
        end
        
        function [X,Y] = nextBatch(self)            
            % Map the indices into data
            miniBatchIndices = self.computeDataIndices();
            
            % Read the data
            [X,Y] = self.readData(miniBatchIndices);
            
            % Advance indices of current mini batch
            self.advanceCurrentMiniBatchIndices();
        end
                    
        function reset(self)
            % reset  Reset iterator state to first batch
            
            self.StartIndexOfCurrentMiniBatch = 1;
        end
        
        function shuffle(self)
            % shuffle  Shuffle the data
            
            self.OrderedIndices = randperm(self.NumberOfObservations);
        end
        
        function reorder(self, indices)
            % reorder   Shuffle the data to a specific order
            self.OrderedIndices = indices;
        end
        
        function set.MiniBatchSize(self,batchSize)
            value = min(batchSize,self.NumberOfObservations);
            self.MiniBatchSizeInternal = min(value,self.NumberOfObservations);
        end
        
        function batchSize = get.MiniBatchSize(self)
            batchSize = self.MiniBatchSizeInternal;
        end
        
    end
    
    methods (Access = private)
        
        function [X,Y] = readData(self,indices)
            % Populate X
            X = self.Input(:,:,:,indices);
            
            % Populate Y
            Y = readResponses(self,indices);
        end
        
        function responses = readResponses(self, indices)
            if isempty(self.Response)
                responses = [];
            else
                if iscategorical(self.Response)
                    % Categorical vector of responses
                    responses = self.Response(indices);
                elseif ismatrix(self.Response)
                    % Matrix of responses
                    responses = iMatrix2Tensor(self.Response(indices,:));
                else
                    % 4D array of responses already in the right shape
                    responses = self.Response(:,:,:,indices);
                end
            end
        end
        
        function dataIndices = computeDataIndices(self)
            % computeDataIndices    Compute the indices into the data from
            % start and end index
            startIdx = min(self.StartIndexOfCurrentMiniBatch,self.NumberOfObservations);
            endIdx = startIdx + self.MiniBatchSize - 1;
            endIdx = min(endIdx,self.NumberOfObservations);
            
            dataIndices = startIdx:endIdx;
            
            % Convert sequential indices to ordered (possibly shuffled) indices
            dataIndices = self.OrderedIndices(dataIndices);
        end
        
        function advanceCurrentMiniBatchIndices(self)
            self.StartIndexOfCurrentMiniBatch = self.StartIndexOfCurrentMiniBatch + self.MiniBatchSize;        
        end
    
    end
end


function tensorResponses = iMatrix2Tensor( matrixResponses )
% iMatrix2Tensor   Convert a matrix of responses of size numObservations x
% numResponses to a tensor of size 1 x 1 x numResponses x numObservations
[numObservations, numResponses] = size( matrixResponses );
tensorResponses = matrixResponses';
tensorResponses = reshape(tensorResponses,[1 1 numResponses numObservations]);
end

function iValidateNumericInputs(input,response)

if isempty(response)
    return % For inference use cases.
end

if iscategorical(response)
   if size(input,4) ~= length(response)
       exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:FourDArrayMiniBatchDatasource:XandYNumObservationsDisagree');
       throwAsCaller(exception);
   end
elseif isnumeric(response)
    if ismatrix(response) && (size(response,1) ~= size(input,4))
        exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:FourDArrayMiniBatchDatasource:XandYNumObservationsDisagree');
        throwAsCaller(exception);
    elseif (ndims(response) == 4) && (size(response,4) ~= size(input,4))
        exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:FourDArrayMiniBatchDatasource:XandYNumObservationsDisagree');
        throwAsCaller(exception);
    elseif (ndims(response) ~= 4) && ~ismatrix(response)
        exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:FourDArrayMiniBatchDatasource:YWrongDimensionality');
        throwAsCaller(exception);
    end
else
    exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:FourDArrayMiniBatchDatasource:UnexpectedTypeProvidedForY');
    throwAsCaller(exception);
end

end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(errorID, getString(message(errorID, varargin{:})));
end