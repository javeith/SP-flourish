function labels = undummify( scores, classNames)
    % undummify   Convert scores into categorical output.
    
    %   Copyright 2015 The MathWorks, Inc.
    
    [~, idx] = max(scores,[],2);
    labels = categorical(classNames(idx));
    % Add the categories back into labels, restoring original ordering
    labels = setcats(labels, classNames);
    labels = iMakeVertical(labels);
end

function vec = iMakeVertical( vec )
    vec = reshape( vec, numel( vec ), 1 );
end