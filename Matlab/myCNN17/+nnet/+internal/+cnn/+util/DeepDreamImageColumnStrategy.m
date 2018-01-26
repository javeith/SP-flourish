% Column strategy for printing verbose log of deepDreamImage.
%
% Copyright 2016 The MathWorks, Inc.
classdef DeepDreamImageColumnStrategy < nnet.internal.cnn.util.ColumnStrategy
   properties
       % HorizontalBorder (char array)   Horizontal border of the table to
        %                                 print
        HorizontalBorder = getString(message('nnet_cnn:deepDreamImage:VerboseHorizBorder'));
        
        % Headings (char array)   Table headings
        Headings = getString(message('nnet_cnn:deepDreamImage:VerboseHeader'));
        
        % Names (cellstr)   Names of Summary properties to be
        %                   reported
        Names = {'Octave', 'Iteration', 'ActivationStrength'};
        
        % Formats (cellstr)   Formats to be used when printing the properties
        Formats = {'%12d', '%12d', '%12.2f'};
   end
    
end
