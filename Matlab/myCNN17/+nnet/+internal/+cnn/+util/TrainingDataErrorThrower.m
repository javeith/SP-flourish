classdef TrainingDataErrorThrower < nnet.internal.cnn.util.ErrorThrowerStrategy
    % TrainingDataErrorThrower   Error thrower strategy for training data
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        % Data validation errors
        function throwImageDatastoreMustHaveCategoricalLabels(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:ImageDatastoreMustHaveCategoricalLabels');
            throwAsCaller(exception);
        end

        function throwImageDatastoreHasNoLabels(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:ImageDatastoreHasNoLabels');
            throwAsCaller(exception);
        end

        function throwXIsNotValidImageArray(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:XIsNotValidImageArray');
            throwAsCaller(exception);
        end

        function throwYIsNotCategoricalResponseVector(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:YIsNotCategoricalResponseVector');
            throwAsCaller(exception);
        end

        function throwYIsNotValidResponseArray(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:YIsNotValidResponseArray');
            throwAsCaller(exception);
        end

        function throwXAndYHaveDifferentObservations(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:XAndYHaveDifferentObservations');
            throwAsCaller(exception);
        end

        function throwXIsNotValidType(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:XIsNotValidType');
            throwAsCaller(exception);
        end

        function throwImageDatastoreWithRegression(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:ImageDatastoreWithRegression');
            throwAsCaller(exception);
        end

        function throwInvalidClassificationTable(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:InvalidClassificationTable');
            throwAsCaller(exception);
        end

        function throwInvalidRegressionTablePredictors(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:InvalidRegressionTablePredictors');
            throwAsCaller(exception);
        end

        function throwInvalidRegressionTableResponses(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:InvalidRegressionTableResponses');
            throwAsCaller(exception);
        end

        function throwUndefinedLabels(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:UndefinedLabels');
            throwAsCaller(exception);
        end
        
        % Data size validation errors
        function throwOutputSizeNumClassesMismatch(~, networkOutputSize, dataNumClasses)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:OutputSizeNumClassesMismatch', networkOutputSize, dataNumClasses);
            throwAsCaller(exception);
        end

        function throwOutputSizeResponseSizeMismatch(~, networkOutputSize, dataResponseSize)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:OutputSizeResponseSizeMismatch', networkOutputSize, dataResponseSize);
            throwAsCaller(exception);
        end

        function throwOutputSizeNumResponsesMismatch(~, networkOutputSize, dataNumResponses)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:OutputSizeNumResponsesMismatch', networkOutputSize, dataNumResponses);
            throwAsCaller(exception);
        end
        
        function throwImagesInvalidSize(~, ImageSize, InputLayerSize)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:ImagesInvalidSize', ImageSize, InputLayerSize);
            throwAsCaller(exception);
        end
        
        function throwSequencesInvalidSize(~, DataSize, InputLayerSize)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:SequencesInvalidSize', DataSize, InputLayerSize);
            throwAsCaller(exception);
        end
        
        function throwXIsNotValidSequenceInput(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:XIsNotValidSequenceInput');
            throwAsCaller(exception);
        end
        
        function throwYIsNotValidSequenceCategorical(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:YIsNotValidSequenceCategorical');
            throwAsCaller(exception);
        end
        
        function throwInvalidResponseSequenceLength(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:InvalidResponseSequenceLength');
            throwAsCaller(exception);
        end
        
        function throwOutputModeLastDataMismatch(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:OutputModeLastDataMismatch');
            throwAsCaller(exception);
        end
        
        function throwOutputModeSequenceDataMismatch(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:OutputModeSequenceDataMismatch');
            throwAsCaller(exception);
        end
        
        function throwIncompatibleInputForLSTM(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:IncompatibleInputForLSTM');
            throwAsCaller(exception);
        end
    end
    
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
errorMessage = getString(message(errorID, varargin{:}));
exception = MException(errorID, errorMessage);
end
