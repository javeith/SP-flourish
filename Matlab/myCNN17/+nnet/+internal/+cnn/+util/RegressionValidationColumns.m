classdef RegressionValidationColumns < nnet.internal.cnn.util.ColumnStrategy
    % RegressionValidationColumns   Regression with validation column
    % strategy
    
    %   Copyright 2017 The MathWorks, Inc.    
    
    properties
        HorizontalBorder = getString(message('nnet_cnn:internal:cnn:RegressionValidationColumns:HorizontalBorder'));
        Headings = getString(message('nnet_cnn:internal:cnn:RegressionValidationColumns:Headings'));
        Names =   { 'Epoch', 'Iteration', 'Time',   'Loss',   'ValidationLoss', 'RMSE',   'ValidationRMSE', 'LearnRate'};
        Formats = { '%12d',  '%12d',      '%12.2f', '%12.4f', '%12.4f',         '%12.2f', '%12.2f',         '%12.4f'};
    end
end