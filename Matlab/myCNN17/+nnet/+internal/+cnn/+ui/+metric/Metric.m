classdef Metric < nnet.internal.cnn.ui.metric.UpdateableMetric
    % Metric   Class which reads information struct and updates its UpdateableSeries if necessary.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Access = private)
        % UpdateableSeries   (nnet.internal.cnn.ui.axes.UpdateableSeries)
        % Series to update 
        UpdateableSeries
        
        % MetricName   (char) Name of struct field name to extract
        MetricName
    end
    
    methods
        function this = Metric(updateableSeries, metricName)
            this.UpdateableSeries = updateableSeries;
            this.MetricName = metricName;
        end
        
        function update(this, infoStruct)
            metricValue = infoStruct.(this.MetricName);
            if iShouldMetricBeUpdated(metricValue)
                xValue = infoStruct.Iteration;
                yValue = infoStruct.(this.MetricName);
                this.UpdateableSeries.add(xValue, yValue); 
            end
        end
    end    
end

function tf = iShouldMetricBeUpdated(metricValue)
tf = ~isempty(metricValue);
end
