%% Description of the Data
%This dataset consists of collection of shape and texture features extracted from digital images of leaf specimens
%originating from a total of 30 different plant species 
%(Description of the dataset says that it has 40 species, but is not the case). 
%The classification goal is to predict the name of species (Label) based on the digital image of the leaf. 
%Dataset has 14 attributes/features but only 340 instances.

%Attributes

% 3. Eccentricity
% 4. Aspect Ratio
% 5. Elongation
% 6. Solidity
% 7. Stochastic Convexity
% 8. Isoperimetric Factor
% 9. Maximal Indentation Depth
% 10. Lobedness
% 11. Average Intensity
% 12. Average Contrast
% 13. Smoothness
% 14. Third moment
% 15. Uniformity
% 16. Entropy

%Output
% 1. Class (Species):  1-30
% 2. Specimen Number ( Total Number of Specimens - Not significant

%% Import Existing Data
% In this example, the data is loaded from the available CSV-file. 
% 
% The data is loaded into dataset arrays. Dataset arrays make it easier to
% work with data of different datatypes to be stored as part of the same
% matrix.

leafData = importLeafData('leaf.csv');
names = leafData.Properties.VarNames;


%% Convert Categorical Data into Nominal Arrays
% Categorical data contains discreet pieces of information, for instance,
% the attribute, marital status in this dataset. 
% Also, dataset arrays allow one to slice the data easily in different ways.
[nrows, ncols] = size(leafData);
category = false(1,ncols);
for i = 1:ncols
    if isa(leafData.(names{i}),'cell') || isa(leafData.(names{i}),'nominal')
        category(i) = true;
        leafData.(names{i}) = nominal(leafData.(names{i}));
    end
end
% Logical array keeping track of categorical attributes
catPred = category(1:end-1);
% Set the random number seed to make the results repeatable in this script
rng('default');

%% Prepare the Data: Response and Predictors
% We can segregate the data into labels and predictors. This will make it
% easier to call subsequent functions which expect the data in this
% format.

% Labels
Y = leafData.Class;
disp('Leaf Classification')
%tabulate(Y)
% Predictor matrix
X = double(leafData(:,3:end));

%% Partition Data Set into Test and Training Set
% The training set will be used to calibrate/train the model parameters.
% The trained model is then used to make a prediction on the test set.
% Predicted values will be compared with actual data to compute the
% confusion matrix. Confusion matrix is one way to visualize the
% performance of a machine learning technique.

% In this example, we will hold 15% of the data, selected randomly, for
% test phase.
cv = cvpartition(length(leafData),'holdout',0.15);

% Training set
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
% Test set
Xtest = X(test(cv),:);
Ytest = Y(test(cv),:);

% disp('Training Set')
% tabulate(Ytrain)
% disp('Test Set')
% tabulate(Ytest)
% 
%% Decision Trees
disp('Leaf Classification - Decision Tree')
tic
t = ClassificationTree.fit(Xtrain,Ytrain);
toc
% Improving Classification Trees

% Examining Resubstitution Error
% Resubstitution error is the difference between the response training data and 
% the predictions the tree makes of the response based on the input training data

resuberror = resubLoss(t)*100

% Cross Validation
% 10 fold cross validation splits the training data into 10 parts at random. 
% It trains 10 new trees, each one on nine parts of the data.
% It then examines the predictive accuracy of each new tree on the data not included in training that tree. 
% This method gives a good estimate of the predictive accuracy of the resulting tree, since it tests the new trees on new data.

cvctree = crossval(t,'KFold',10);
cvloss = kfoldLoss(cvctree)*100


% Make a prediction for the test set
Y_t = t.predict(Xtest);
Y_trainError = t.predict(Xtrain);

% Compute the confusion matrix
C_t_train = confusionmat(Ytrain,Y_trainError);
C_t_test = confusionmat(Ytest,Y_t);

N = sum(C_t_test(:));
Test_err = (( N-sum(diag(C_t_test)) ) / N )*100

% Examine the confusion matrix for each class as a percentage of the true class
C_t_train = bsxfun(@rdivide,C_t_train,sum(C_t_train,2)) * 100;
C_t_test = bsxfun(@rdivide,C_t_test,sum(C_t_test,2)) * 100;

%Controlling the tree depth

%Control Depth or "Leafiness"
% Following code was used to Find the optimal Control Depth. 
% Uncomment to run and generate the graph
% Generate minimum leaf occupancies for classification trees from 10 to 1000, spaced exponentially apart:

% leafs = logspace(0,1,10);
% rng('default')
% N = numel(leafs);
% err = zeros(N,1);
% for n=1:N
%     ctree = ClassificationTree.fit(Xtrain,Ytrain,'CrossVal','On',...
%         'MinLeaf',leafs(n));
%     err(n) = kfoldLoss(ctree);
% end
% plot(leafs,err);
% xlabel('Min Leaf Size');
% ylabel('cross-validated error');

%%Optimal Tree using Control Depth
%%The best leaf size is between about 2-6 observations per leaf.

%%DefaultTree
% view(t,'Mode','Graph')
% 
disp('Optimal Tree using Leafiness')
OptimalTree = ClassificationTree.fit(Xtrain,Ytrain,'minleaf',2);
view(OptimalTree,'mode','graph')

resubOpt = resubLoss(OptimalTree)*100
cvlossOpt = kfoldLoss(crossval(OptimalTree))*100

% Make a prediction for the test set
Y_t = OptimalTree.predict(Xtest);
Y_trainError = OptimalTree.predict(Xtrain);

% Compute the confusion matrix
C_t_train = confusionmat(Ytrain,Y_trainError);
C_t_test = confusionmat(Ytest,Y_t);

N = sum(C_t_test(:));
Test_err_Optimal = (( N-sum(diag(C_t_test)) ) / N )*100

disp('Confusion Matrix')
C_t_train = bsxfun(@rdivide,C_t_train,sum(C_t_train,2)) * 100;
C_t_test = bsxfun(@rdivide,C_t_test,sum(C_t_test,2)) * 100;

%%Pruning
%Pruning optimizes tree depth (leafiness) is by merging leaves on the same tree branch
%Unlike in that section, you do not need to grow a new tree for every node size. 
%Instead, grow a deep tree, and prune it to the level you choose.
disp('Optimal Tree using Pruning')
[~,~,~,bestlevel] = cvLoss(t,'SubTrees','All','TreeSize','min');
PrunedTree = prune(t,'Level',bestlevel);
%view(PrunedTree,'Mode','Graph')

resubPruned = resubLoss(PrunedTree)*100
cvlossPruned = kfoldLoss(crossval(PrunedTree))*100

% Make a prediction for the test set
Y_t = PrunedTree.predict(Xtest);
Y_trainError = PrunedTree.predict(Xtrain);

% Compute the confusion matrix
C_t_train = confusionmat(Ytrain,Y_trainError);
C_t_test = confusionmat(Ytest,Y_t);

N = sum(C_t_test(:));
Test_err_Pruned = (( N-sum(diag(C_t_test)) ) / N )*100

disp('Confusion Matrix')
C_t_train = bsxfun(@rdivide,C_t_train,sum(C_t_train,2)) * 100;
C_t_test = bsxfun(@rdivide,C_t_test,sum(C_t_test,2)) * 100;

%% Prepare Predictors/Response for Neural Networks
% When using neural networks the appropriate way to include categorical
% predictors is as dummy indicator variables. An indicator variable has
% values 0 and 1.

[XtrainNN, YtrainNN, XtestNN, YtestNN] = prepare_NN_leaf(leafData, cv);

%% Speed up Computations using Parallel Computing
% If Parallel Computing Toolbox is available, the computation will be
% distributed to 2 workers for speeding up the evaluation.

if matlabpool('size') == 0 
    matlabpool open 2
end

%% Neural Networks
% Neural Network Toolbox supports supervised learning with feedforward,
% radial basis, and dynamic networks. It supports both classification and
% regression algorithms. It also supports unsupervised learning with
% self-organizing maps and competitive layers.
% 
% One can make use of the interactive tools to setup, train and validate a
% neural network. It is then possible to auto-generate the code for the
% purpose of automation. In this example, the auto-generated code has been
% updated to utilize a pool of workers, if available. This is achieved by
% simply setting the _useParallel_ flag while making a call to |train|.
% 
%   [net,~] = train(net,inputs,targets,'useParallel','yes');
% 
% If a GPU is available, it may be utilized by setting the _useGPU_ flag.
% 
% The trained network is used to make a prediction on the test data and
% confusion matrix is generated for comparison with other techniques.

% In order to change Learning Rate, Momentum, Hidden Layer size Check NNfun
disp('Leaf Classification- Neural Networks')
[~, net] = NNfun(XtrainNN,YtrainNN);

% Make a prediction for the test set
Y_nn_train = net(XtrainNN');
Y_nn_train = round(Y_nn_train');

Y_nn_test = net(XtestNN');
Y_nn_test = round(Y_nn_test');

% Compute the confusion matrix
Y_nn_train = vec2ind(Y_nn_train');
YtrainNN = vec2ind(YtrainNN');
Y_nn_test = vec2ind(Y_nn_test');
YtestNN = vec2ind(YtestNN');
C_nn_train = confusionmat(YtrainNN,Y_nn_train);
C_nn_test = confusionmat(YtestNN,Y_nn_test);
N = sum(C_nn_train(:));
Training_err =( ( N-sum(diag(C_nn_train)) ) / N )* 100
N = sum(C_nn_test(:));
Test_err = (( N-sum(diag(C_nn_test)) ) / N )*100
% Examine the confusion matrix for each class as a percentage of the true class
C_nn_test = bsxfun(@rdivide,C_nn_test,sum(C_nn_test,2)) * 100;
C_nn_train = bsxfun(@rdivide,C_nn_train,sum(C_nn_train,2)) * 100;

%% K Nearest Neighbors

disp('Leaf Classification - K Nearest Neighbor')
% Train the classifier
%knn = ClassificationKNN.fit(Xtrain,Ytrain,'Distance','seuclidean','NumNeighbors',2);
K = 1:1:40;
rng('default')
N = numel(K);
err = zeros(N,1);
for n=1:N
    %Change the distance metric to 'euclidean' for euclidean, 'Chebychev' for Chebychev, and 'seuclidean' for standard euclidean
    knn = ClassificationKNN.fit(Xtrain,Ytrain,'Distance','seuclidean','DistanceWeight','inverse','NumNeighbors',K(n));
    knnRloss(n) = resubLoss(knn);
    err(n) = kfoldLoss(crossval(knn));
    Y_knn = knn.predict(Xtest);
    C_knn_test = confusionmat(Ytest,Y_knn);
    N1 = sum(C_knn_test(:));
Test_err(n) = (( N1-sum(diag(C_knn_test)) ) / N1 );
end
plot(K,err);
hold on
plot(K, knnRloss,'r');
hold on
plot(K, Test_err,'g');
title('K-NN');
xlabel('Number of Neighbors');
ylabel('Error');
legend('Cross Validation Error','Training error','Test Error','location','SE');

%% Boosting

disp('Leaf Classification - Boosting')
%Using the same unpruned tree
bag = fitensemble(Xtrain,Ytrain,'Bag',200,'Tree',...
    'Type','Classification');
%using the same cv as for all algorithm
cv1 = fitensemble(X,Y,'Bag',200,'Tree',...
    'type','classification','kfold',10);
figure;
plot(loss(bag,Xtest,Ytest,'mode','cumulative'),'g');
hold on;
plot(kfoldLoss(cv1,'mode','cumulative'));
hold on
plot(loss(bag,Xtrain,Ytrain,'mode','cumulative'),'r');
hold off;
title('Boosting');
xlabel('Number of trees');
ylabel('Classification error');
legend('Test','Cross-validation','Training','Location','NE');
Ensemble = fitensemble(Xtrain,Ytrain,'Bag',200,'Tree','Type','Classification');
Y_b_test  = Ensemble.predict(Xtest);
Y_b_train  = Ensemble.predict(Xtrain);

C_b_train = confusionmat(Ytrain,Y_b_train);
C_b_test = confusionmat(Ytest,Y_b_test);
N = sum(C_b_train(:));
Training_err =( ( N-sum(diag(C_b_train)) ) / N )* 100
N = sum(C_b_test(:));
Test_err = (( N-sum(diag(C_b_test)) ) / N )*100

% Examine the confusion matrix for each class as a percentage of the true class
C_b_test = bsxfun(@rdivide,C_b_test,sum(C_b_test,2)) * 100;
C_b_train = bsxfun(@rdivide,C_b_train,sum(C_b_train,2)) * 100;

%%  Support Vector Machines

disp('Leaf Classification - SVM')
% Inorder to change kernel and its parameters, check multisvm.m file

Y_SVM_test = multisvm(Xtrain, Ytrain, Xtest);

Y_SVM_train = multisvm(Xtrain, Ytrain, Xtrain);



%Compute the confusion matrix
C_SVM_train = confusionmat(Ytrain,Y_SVM_train);
C_SVM_test = confusionmat(Ytest,Y_SVM_test);
N = sum(C_SVM_train(:));
Training_err =( ( N-sum(diag(C_SVM_train)) ) / N )* 100
N = sum(C_SVM_test(:));
Test_err = (( N-sum(diag(C_SVM_test)) ) / N )*100
%Examine the confusion matrix for each class as a percentage of the true class
C_SVM_test = bsxfun(@rdivide,C_SVM_test,sum(C_SVM_test,2)) * 100;
C_SVM_train = bsxfun(@rdivide,C_SVM_train,sum(C_SVM_train,2)) * 100;

%% Shut Down Workers
% % Release the workers if there is no more work for them

if matlabpool('size') > 0
    matlabpool close
end