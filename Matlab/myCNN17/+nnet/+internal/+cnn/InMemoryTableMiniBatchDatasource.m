classdef InMemoryTableMiniBatchDatasource <...
        nnet.internal.cnn.MiniBatchDatasource &...
        nnet.internal.cnn.NamedResponseMiniBatchDatasource &...
        nnet.internal.cnn.DistributableInMemoryTableMiniBatchDatasource &...
        nnet.internal.cnn.BackgroundDispatchableDatasource

    % InMemoryTableMiniBatchDatasource class to extract 4D data one mini
    % batch at a time from a table
    %
    % Input data    - a table containing predictors and responses. The
    %               first column will contain predictors in form of cell
    %               array of images. Responses will be held in the second
    %               column as either vectors or cell arrays containing 3-D
    %               arrays or categorical labels. Alternatively, responses
    %               will be held in multiple columns as scalars.
    % Output data   - 4D data where the fourth dimension is the number of
    %               observations in that mini batch.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Dependent)
       MiniBatchSize 
    end
    
    properties
        NumberOfObservations
        ResponseNames
    end
    
    properties (Access = ?nnet.internal.cnn.DistributableMiniBatchDatasource)
        % Table   (Table) The table containing all the data
        TableData
    end
        
    properties (Access = private)
        StartIndexOfCurrentMiniBatch 
        OrderedIndices % Shuffled sequence of all indices into observations
        MiniBatchSizeInternal
    end
    
    methods
        
        function self = InMemoryTableMiniBatchDatasource(tableIn,miniBatchSize)
            
            if ~isempty(tableIn)
                self.TableData = tableIn;
                self.ResponseNames = iResponseNamesFromTable(tableIn);
                self.NumberOfObservations = size(tableIn,1);
                self.OrderedIndices = 1:self.NumberOfObservations;
                self.MiniBatchSize = miniBatchSize;
                self.reset();
            else
                self = nnet.internal.cnn.InMemoryTableMiniBatchDatasource.empty();
            end
        end
                
        function [X,Y] = getObservations(self,indices)
            % getObservations  Overload of method to retrieve specific
            % observations.
            
            [X,Y] = self.readData(self.OrderedIndices(indices));
        end
        
        function [X,Y] = nextBatch(self)
            % nextBatch  Return next mini-batch of data
            
            % Map the indices into data
            miniBatchIndices = self.computeDataIndices();
            
            % Read the data
            [X,Y] = self.readData(miniBatchIndices);
            
            % Advance indices of current mini batch
            self.advanceCurrentMiniBatchIndices();
        end
                    
        function reset(self)
            % reset  Reset iterator state to first mini-batch
            
            self.StartIndexOfCurrentMiniBatch = 1;
        end
        
        function shuffle(self)
            % shuffle  Shuffle the data
            
            self.OrderedIndices = randperm(self.NumberOfObservations);
        end
        
        function reorder(self,indices)
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
            if isempty(self.TableData)
                [X,Y] = deal([]);
            else
                X = self.readInput(indices);
                Y = self.readResponses(indices);
            end
        end
        
        function response = readResponses(self,indices)
            
            singleResponseColumn = size(self.TableData,2) == 2;
            if singleResponseColumn
               response = self.TableData{indices,2};
               if isvector(self.TableData(1,2))
                  response = iMatrix2Tensor(response);
               end
            else
               response = iMatrix2Tensor(self.TableData{indices,2:end}); 
            end
            
        end
        
        function X = readInput(self,indices)
           
            X = self.TableData{indices,1};
            if any(cellfun(@isempty,X))
               iThrowEmptyImageDataException(); 
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

function iThrowEmptyImageDataException()
% iThrowEmptyImageDataException   Throw an empty image data exception
exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:TableMiniBatchDatasource:EmptyImageData');
throwAsCaller(exception)
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(errorID, getString(message(errorID, varargin{:})));
end

function tensorResponses = iMatrix2Tensor( matrixResponses )
% iMatrix2Tensor   Convert a matrix of responses of size numObservations x
% numResponses to a tensor of size 1 x 1 x numResponses x numObservations
[numObservations, numResponses] = size( matrixResponses );
tensorResponses = matrixResponses';
tensorResponses = reshape(tensorResponses,[1 1 numResponses numObservations]);
end

function responseNames = iResponseNamesFromTable( tableData )
responseNames = tableData.Properties.VariableNames(2:end);
% To be consistent with ClassNames, return a column array
responseNames = responseNames';
end