classdef ClassificationValidationColumns < nnet.internal.cnn.util.ColumnStrategy
    % ClassificationValidationColumns   Classification with validation
    % column strategy
    
    %   Copyright 2017 The MathWorks, Inc.    
    
    properties
        HorizontalBorder = getString(message('nnet_cnn:internal:cnn:ClassificationValidationColumns:HorizontalBorder'));
        Headings = getString(message('nnet_cnn:internal:cnn:ClassificationValidationColumns:Headings'));
        Names =   { 'Epoch', 'Iteration', 'Time',   'Loss',   'ValidationLoss', 'Accuracy', 'ValidationAccuracy', 'LearnRate'};
        Formats = { '%12d',  '%12d',      '%12.2f', '%12.4f', '%12.4f',         '%11.2f%%', '%11.2f%%',           '%12.4f'};
    end
end