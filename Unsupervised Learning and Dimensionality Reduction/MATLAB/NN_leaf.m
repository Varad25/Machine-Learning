clear all;
close all;
%% Loading Data

load YNN.mat;

% Original
inputs = X';
targets = YNN';
%% PCA
% dim = 4;
% K=25;
% [COEFF, SCORE, LATENT] = pca(X,'NumComponents',dim);
% %
% inputs = SCORE';
% targets = YNN';

%% ICA

%  [z_ic A T mean_z] = myICA(X',4);
%  inputs = z_ic;
% targets = YNN';'

%% RP
% load z_rp_min.mat;
% inputs = z_rp_min;
% targets = YNN';

%% Clustering as Dimension Reduction Algorithm
% PCA
% dim = 4;
% k=30;
% [COEFF, SCORE, LATENT] = pca(X,'NumComponents',dim);
% % K-Means
% %[id_cl_k,C] = kmeans(SCORE,k,'Distance','sqeuclidean','Display','final');
%  %inputs = id_cl_k';
%  
%  % EM
%  [id_pca_em,model, llh]= emgm(SCORE',k);
% inputs = id_pca_em;
% 
% targets = YNN';

% ICA
% k= 30;
%  [z_ic A T mean_z] = myICA(X',4);
%  % K-means
%  
% %  [id_ica,C] = kmeans(z_ic',k,'Distance','sqeuclidean','Display','final');
% % inputs = id_ica';
% 
%  
%  % EM
%  [id_ica_em,model, llh]= emgm(z_ic,k);
%  inputs = id_ica_em;
%  
%   targets = YNN';

%% K - Means
% k = 30;
% [id_k,C] = kmeans(X,k,'Distance','sqeuclidean');
% inputs = [ X id_k]';
% targets = YNN';

%% EM
% k= 30;
% [id_em,model, llh]= emgm(X',k);
% inputs = [X'; id_em];
% targets =YNN';

%% Neural Network Training

% Create a Pattern Recognition Network
hiddenLayerSize = 50;
net = patternnet([hiddenLayerSize hiddenLayerSize hiddenLayerSize hiddenLayerSize hiddenLayerSize hiddenLayerSize]);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};


% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% For help on training function 'trainscg' type: help trainscg
% For a list of all training functions type: help nntrain
% Train the Network
net.trainFcn = 'traingdm';
net.trainParam.epochs = 20000;
%Learning Rate
net.trainParam.lr = 0.4;
%Momentum
net.trainParam.mc = 0.4;
% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean squared error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
% net.plotFcns = {'plotperform','plottrainstate','ploterrhist', 'plotregression', 'plotfit'};

tic
% Train the Network
[net,tr] = train(net,inputs,targets);
toc

% Test the Network
tic
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)
toc
% Recalculate Training, Validation and Test Performance
trainTargets = targets .* tr.trainMask{1};
valTargets = targets  .* tr.valMask{1};
testTargets = targets  .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,outputs)
valPerformance = perform(net,valTargets,outputs)
testPerformance = perform(net,testTargets,outputs)

% View the Network
%view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, plotconfusion(targets,outputs)
%figure, plotroc(targets,outputs)
%figure, ploterrhist(errors)
