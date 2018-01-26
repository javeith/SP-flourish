classdef ImageDatastoreMiniBatchDatasource <...
        nnet.internal.cnn.MiniBatchDatasource &...
        nnet.internal.cnn.DistributableImageDatastoreMiniBatchDatasource &...
        nnet.internal.cnn.BackgroundDispatchableDatasource
     
    % ImageDatastoreMiniBatchDatasource class to extract data one
    % mini-batch at a time from an imageDatastore.
    %
    % Input data    - an image datastore containing either RGB or grayscale
    %               images of the same size
    % Output data   - 4D data where the fourth dimension is the number of
    %               observations in that mini batch. If the input is a
    %               grayscale image then the 3rd dimension will be 1
    
    %   Copyright 2017 The MathWorks, Inc.
    
   properties
      NumberOfObservations
   end
   
   properties (Dependent)
       MiniBatchSize
   end
      
   properties (Access = ?nnet.internal.cnn.DistributableMiniBatchDatasource)
       imds
   end
   
   methods
             
       function self = ImageDatastoreMiniBatchDatasource(imds,miniBatchSize)
           
           if ~isempty(imds)
               self.imds = imds.copy(); % Don't mess with state of imds input.
               self.NumberOfObservations = length(self.imds.Files);
               miniBatchSize = min(miniBatchSize,self.NumberOfObservations);
               self.MiniBatchSize = miniBatchSize;
               
               % Only automatically run in background if the datastore has a
               % custom ReadFcn, otherwise it will likely be slower than the
               % threaded prefetch used automatically
               if nnet.internal.cnn.util.imdsHasCustomReadFcn(self.imds)
                   self.RunInBackgroundOnAuto = true;
               end
               
               self.reset();
           else
              self = nnet.internal.cnn.ImageDatastoreMiniBatchDatasource.empty(); 
           end
           
       end
              
       function [X,Y] = getObservations(self,indices)
           % getObservations  Overload of method to retrieve specific
           % observations. The implementation for image datastore is
           % inefficient so should only be used when cost of dispatch is
           % masked (because it happens in the background for instance).
           
           % Populate Y
           if isempty(self.imds.Labels)
               Y = [];
           else
               Y = self.imds.Labels(indices);
           end
           
           % Create datastore partition via a copy and index. This is
           % faster than constructing a new datastore with the new
           % files.
           subds = copy(self.imds);
           subds.Files = self.imds.Files(indices);
           X = subds.readall();
       end
       
       function [X,Y] = nextBatch(self)
           % nextBatch  Return next mini-batch
           
           [X,metaStruct] = self.imds.read();
           Y = metaStruct.Label;
       end
       
       function reset(self)
           % reset  Reset iterator state to first mini-batch
           
          self.imds.reset();
       end
              
       function shuffle(self)
           % shuffle  Shuffle data           
           if ~isempty(self.imds)
               self.imds = self.imds.shuffle();
           end
       end
       
       function reorder(self, indices)
           % reorder   Shuffle data to a specific order
           
           if ~isempty(self.imds)
               newDatastore = copy(self.imds);
               newDatastore.Files = self.imds.Files(indices);
               newDatastore.Labels = self.imds.Labels(indices);
               self.imds = newDatastore;
           end
       end
       
       function batchSize = get.MiniBatchSize(self)
           batchSize = self.imds.ReadSize;
       end
       
       function set.MiniBatchSize(self,batchSize)
           self.imds.ReadSize = batchSize;
       end
       
   end
end
