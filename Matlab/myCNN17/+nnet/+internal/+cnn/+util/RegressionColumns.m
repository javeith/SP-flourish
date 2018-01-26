classdef RegressionColumns < nnet.internal.cnn.util.ColumnStrategy
    % RegressionColumns   Regression column strategy
    
    %   Copyright 2016 The MathWorks, Inc.    
    
    properties
        HorizontalBorder = getString(message('nnet_cnn:internal:cnn:RegressionColumns:HorizontalBorder'));
        Headings = getString(message('nnet_cnn:internal:cnn:RegressionColumns:Headings'));
        Names = {'Epoch','Iteration','Time','Loss','RMSE','LearnRate'};
        Formats = { '%12d', '%12d', '%12.2f', '%12.4f', '%12.2f', '%12.4f'};
    end
end