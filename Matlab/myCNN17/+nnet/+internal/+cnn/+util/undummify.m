function labels = undummify( scores, classNames)
    % undummify   Convert scores into categorical output. 
    % classNames is supposed to be a column vector of categories. The
    % underlying categories must already be in the correct order. 
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    [mxValues, idx] = max(scores,[],2);
    labels = classNames(idx);
    
    % Replace NaN maxima with <undefined> labels
    nans = isnan(mxValues);
    if any(nans)
        labels(nans) = categorical(NaN);
    end
end