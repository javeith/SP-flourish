function layer = lstmLayer(varargin)
%lstmLayer   Long Short-Term Memory (LSTM) layer
%
%   layer = lstmLayer(outputSize) creates a Long Short-Term Memory layer.
%   outputSize is the size of the output dimensions of the layer, specified
%   as a positive integer.
%
%   layer = lstmLayer(outputSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%     'Name'                              - Name for the layer, specified
%                                           as a character vector. The
%                                           default value is ''.
%     'OutputMode'                        - The format of the output of the
%                                           layer. Options are:
%                                               - 'sequence', to output a
%                                               full sequence.
%                                               - 'last', to output the
%                                               last element only. 
%                                           The default value is
%                                           'sequence'.
%     'InputWeightsLearnRateFactor'       - Multiplier for the learning 
%                                           rate of the input weights,
%                                           specified as a scalar or a
%                                           1-by-4 row vector. The default
%                                           value is 1.
%     'RecurrentWeightsLearnRateFactor'   - Multiplier for the learning 
%                                           rate of the recurrent weights,
%                                           specified as a scalar or a
%                                           1-by-4 row vector. The default
%                                           value is 1.
%     'BiasLearnRateFactor'               - Multiplier for the learning 
%                                           rate of the bias, specified as
%                                           a scalar or a 1-by-4 row
%                                           vector. The default value is 1.
%     'InputWeightsL2Factor'              - Multiplier for the L2
%                                           regularizer of the input
%                                           weights, specified as a scalar
%                                           or a 1-by-4 row vector. The
%                                           default value is 1.
%     'RecurrentWeightsL2Factor'          - Multiplier for the L2
%                                           regularizer of the recurrent
%                                           weights, specified as a scalar
%                                           or a 1-by-4 row vector. The
%                                           default value is 1.
%     'BiasL2Factor'                      - Multiplier for the L2
%                                           regularizer of the bias,
%                                           specified as a scalar or a
%                                           1-by-4 row vector. The default
%                                           value is 1.
%
%   Example 1:
%       Create an LSTM layer with 100 output units, with stateful
%       propagation of the cell state and output state.
%
%       layer = lstmLayer(100);
%
%   Example 2:
%       Create an LSTM layer with 50 output units which returns a single
%       element. Manually initialize the recurrent weights and the initial
%       output state from a Gaussian with standard deviation 0.01
%
%       outputSize = 50;
%       layer = lstmLayer(outputSize, 'OutputMode', 'last');
%       layer.RecurrentWeights = randn([4*outputSize outputSize])*0.01;
%
%   See also nnet.cnn.layer.LSTMLayer

%   Copyright 2017 The MathWorks, Inc.

% Parse the input arguments.
args = nnet.cnn.layer.LSTMLayer.parseInputArguments(varargin{:});

% Create an internal representation of the layer.
internalLayer = nnet.internal.cnn.layer.LSTM(args.Name, ...
    args.InputSize, ...
    args.OutputSize, ...
    true, ...
    true, ...
    iGetReturnSequence(args.OutputMode));

% Use the internal layer to construct a user visible layer.
layer = nnet.cnn.layer.LSTMLayer(internalLayer);

% Set learn rate and L2 Factors.
layer.InputWeightsL2Factor = args.InputWeightsL2Factor;
layer.InputWeightsLearnRateFactor = args.InputWeightsLearnRateFactor;

layer.RecurrentWeightsL2Factor = args.RecurrentWeightsL2Factor;
layer.RecurrentWeightsLearnRateFactor = args.RecurrentWeightsLearnRateFactor;

layer.BiasL2Factor = args.BiasL2Factor;
layer.BiasLearnRateFactor = args.BiasLearnRateFactor;

end

function tf = iGetReturnSequence( mode )
tf = true;
if strcmp( mode, 'last' )
    tf = false;
end
end