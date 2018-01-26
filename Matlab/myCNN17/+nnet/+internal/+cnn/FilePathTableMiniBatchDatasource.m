classdef FilePathTableMiniBatchDatasource <...
        nnet.internal.cnn.MiniBatchDatasource &...
        nnet.internal.cnn.NamedResponseMiniBatchDatasource &...
        nnet.internal.cnn.DistributableFilePathTableMiniBatchDatasource &...
        nnet.internal.cnn.BackgroundDispatchableDatasource

    % FilePathTableMiniBatchDatasource  class to extract 4D data from table data
    %
    % Input data    - a table containing predictors and responses. The
    %               first column will contain predictors holding file paths
    %               to the images. Responses will be held in the second
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
        Datastore
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
        
        function self = FilePathTableMiniBatchDatasource(tableIn,miniBatchSize)
            self.TableData = tableIn;
            
            if ~isempty(tableIn)
                self.Datastore = iCreateDatastoreFromTable(tableIn);
                self.ResponseNames = iResponseNamesFromTable(tableIn);
                self.NumberOfObservations = size(tableIn,1);
                self.OrderedIndices = 1:self.NumberOfObservations;
                self.MiniBatchSize = miniBatchSize;
                self.reset();
            else
                self = nnet.internal.cnn.FilePathTableMiniBatchDatasource.empty();
            end
        end
                
        function [X,Y] = getObservations(self,indices)
            % getObservations  Overload of method to retrieve specific
            % observations.
            
            Y = self.readResponses(self.OrderedIndices(indices));
            
            % Create datastore partition via a copy and index. This is
            % faster than constructing a new datastore with the new
            % files.
            subds = copy(self.Datastore);
            subds.Files = self.Datastore.Files(indices);
            X = subds.readall();
            
        end
                
        function [X,Y] = nextBatch(self)
            % nextBatch  Return next mini-batch
            
            % Map the indices into data
            miniBatchIndices = self.computeDataIndices();
            
            % Read the data
            [X,Y] = self.readData(miniBatchIndices);
            
            % Advance indices of current mini batch
            self.advanceCurrentMiniBatchIndices();
        end
                    
        function reset(self)
            % Reset iterator state to first mini-batch
            
            self.StartIndexOfCurrentMiniBatch = 1;
            self.Datastore.reset();
        end
        
        function shuffle(self)
            % Shuffle  Shuffle the data
            
            self.OrderedIndices = randperm(self.NumberOfObservations);
            self.Datastore = iCreateDatastoreFromTable(self.TableData,self.OrderedIndices);
        end
        
        function reorder(self,indices)
            % reorder   Shuffle the data to a specific order
            
            self.OrderedIndices = indices;
            self.Datastore = iCreateDatastoreFromTable(self.TableData,self.OrderedIndices);
        end
        
        function set.MiniBatchSize(self,batchSize)
            value = min(batchSize,self.NumberOfObservations);
            self.MiniBatchSizeInternal = value;
            if ~isempty(self.TableData)
                self.Datastore.ReadSize = max(1,value);
            end
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
                X = self.Datastore.read();
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

function dataStore = iCreateDatastoreFromTable( aTable, shuffleIdx )

% Assume the first column of the table contains the paths to the images
if nargin < 2
    filePaths = aTable{:,1}'; % 1:end
else
    filePaths = aTable{shuffleIdx,1}'; % Specific shuffle order
end

if any( cellfun(@isdir,filePaths) )
    % Directories are not valid paths
    iThrowWrongImagePathException();
end
try
    dataStore = imageDatastore( filePaths );
catch e
    iThrowFileNotFoundAsWrongImagePathException(e);
    iThrowInvalidStrAsEmptyPathException(e);
    rethrow(e)
end
numObservations = size( aTable, 1 );
numFiles = numel( dataStore.Files );
if numFiles ~= numObservations
    % If some files were discarded when the datastore was created, those
    % files were not valid images and we should error out
    iThrowWrongImagePathException();
end
end

function iThrowWrongImagePathException()
% iThrowWrongImagePathException   Throw a wrong image path exception
exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:TableMiniBatchDatasource:WrongImagePath');
throwAsCaller(exception)
end

function iThrowFileNotFoundAsWrongImagePathException(e)
% iThrowWrongImagePathException   Throw a
% MATLAB:datastoreio:pathlookup:fileNotFound as a wrong image path
% exception.
if strcmp(e.identifier,'MATLAB:datastoreio:pathlookup:fileNotFound')
    iThrowWrongImagePathException()
end
end

function iThrowInvalidStrAsEmptyPathException(e)
% iThrowInvalidStrAsEmptyPathException   Throws a
% pathlookup:invalidStrOrCellStr exception as a EmptyImagePaths exception
if (strcmp(e.identifier,'MATLAB:datastoreio:pathlookup:invalidStrOrCellStr'))
    exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:TableMiniBatchDatasource:EmptyImagePaths');
    throwAsCaller(exception)
end
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(errorID, getString(message(errorID, varargin{:})));
end

function responseNames = iResponseNamesFromTable( tableData )
responseNames = tableData.Properties.VariableNames(2:end);
% To be consistent with ClassNames, return a column array
responseNames = responseNames';
end

function tensorResponses = iMatrix2Tensor( matrixResponses )
% iMatrix2Tensor   Convert a matrix of responses of size numObservations x
% numResponses to a tensor of size 1 x 1 x numResponses x numObservations
[numObservations, numResponses] = size( matrixResponses );
tensorResponses = matrixResponses';
tensorResponses = reshape(tensorResponses,[1 1 numResponses numObservations]);
end