function [outputs, net] = NNfun(X,Y)
% Solve a Pattern Recognition Problem with a Neural Network

%
% This script assumes these variables are defined:
%
%   X - input data.
%   Y - target data.


inputs = X';
targets = Y';
%To avoid Random results
setdemorandstream(391418381);
% Create a Pattern Recognition Network
% For Bank, Hidden Layers = 1 and size of 1 layer = 5
% For Leaf, Hidden Layers = 5, and size of 1 layer = 50

%For Bank Marketing
% hiddenLayerSize = 5;
% net = patternnet([hiddenLayerSize]);

%For Leaf
hiddenLayerSize = 50;
net = patternnet([hiddenLayerSize hiddenLayerSize hiddenLayerSize hiddenLayerSize hiddenLayerSize hiddenLayerSize]);


% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 30/100;

tic
% Train the Network
net.trainFcn = 'traingdm';
net.trainParam.epochs = 20000;
%Learning Rate
net.trainParam.lr = 0.2;
%Momentum
net.trainParam.mc = 0.2;
% Hidden the training window
net.trainParam.showWindow = false;
% Making use of parallel computing
[net,tr] = train(net,inputs,targets,'useParallel','yes');
toc
tic
% Test the Network
outputs = net(inputs);
% errors = gsubtract(targets,outputs);
% performance = perform(net,targets,outputs)
toc
% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
figure, plotperform(tr)
 %figure, plottrainstate(tr)
% figure, plotconfusion(targets,outputs)
% figure, plotroc(targets,outputs)
% figure, ploterrhist(errors)
