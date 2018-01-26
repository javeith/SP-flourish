classdef ClassificationColumns < nnet.internal.cnn.util.ColumnStrategy
    % ClassificationColumns   Classification column strategy
    
    %   Copyright 2016 The MathWorks, Inc.    
    
    properties
        HorizontalBorder = getString(message('nnet_cnn:internal:cnn:ClassificationColumns:HorizontalBorder'));
        Headings = getString(message('nnet_cnn:internal:cnn:ClassificationColumns:Headings'));
        Names = {'Epoch','Iteration','Time','Loss','Accuracy','LearnRate'};
        Formats = { '%12d', '%12d', '%12.2f', '%12.4f', '%11.2f%%', '%12.4f'};
    end
end