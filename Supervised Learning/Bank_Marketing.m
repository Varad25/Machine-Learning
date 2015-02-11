%% Description of the Data
% A publicly available dataset is utilized. It's details are described in
% [Moro et al., 2011]. The data is related with direct marketing campaigns
% of a Portuguese banking institution. The marketing campaigns were based
% on phone calls. Often, more than one contact to the same client was
% required in order to assess if the product (bank term deposit) would be
% or would not be subscribed.
% 
% The classification goal is to predict if the client will subscribe a term
% deposit or not (variable y). The data set contains 45211 observations
% capturing 16 attributes/features.
% 
% Attributes:
% 
% # age (numeric)
% # job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
%  "blue-collar","self-employed","retired","technician","services") 
% # marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
% # education (categorical: "unknown","secondary","primary","tertiary")
% # default: has credit in default? (binary: "yes","no")
% # balance: average yearly balance, in euros (numeric) 
% # housing: has housing loan? (binary: "yes","no")
% # loan: has personal loan? (binary: "yes","no")
% # contact: contact communication type (categorical: "unknown","telephone","cellular") 
% # day: last contact day of the month (numeric)
% # month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
% # duration: last contact duration, in seconds (numeric)
% # campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
% # pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
% # previous: number of contacts performed before this campaign and for this client (numeric)
% # poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
% 
% Output variable (desired target):
% 
% # y: has the client subscribed a term deposit? (binary: "yes","no")

%% Import Existing Data
% In this example, the data is loaded from the available CSV-file. 
% 
% The data is loaded into dataset arrays. Dataset arrays make it easier to
% work with data of different datatypes to be stored as part of the same
% matrix.

bank = ImportBankData('bank-full.csv');
names = bank.Properties.VarNames;

%% Convert Categorical Data into Nominal Arrays
% Categorical data contains discreet pieces of information, for instance,
% the attribute, marital status in this dataset. 
% Also, dataset arrays allow one to slice the data easily in different ways.

% Remove unnecessary double quotes from certain attributes
bank = datasetfun(@removequotes,bank,'DatasetOutput',true);

% Convert all the categorical variables into nominal arrays
[nrows, ncols] = size(bank);
category = false(1,ncols);
for i = 1:ncols
    if isa(bank.(names{i}),'cell') || isa(bank.(names{i}),'nominal')
        category(i) = true;
        bank.(names{i}) = nominal(bank.(names{i}));
    end
end
% Logical array keeping track of categorical attributes
catPred = category(1:end-1);
% Set the random number seed to make the results repeatable in this script
rng('default');

%% Prepare the Data: Response and Predictors
% We can segregate the data into labels and predictors. This will make it
% easier to call subsequent functions which expect the data in this format.

% Labels
Y = bank.y;
disp('Bank Marketing Campaign')
tabulate(Y)
% Predictor matrix
X = double(bank(:,1:end-1));

%% Partition Data Set into Test and Training Set
% The training set will be used to calibrate/train the model parameters.
% The trained model is then used to make a prediction on the test set.
% Predicted values will be compared with actual data to compute the
% confusion matrix. Confusion matrix is one way to visualize the
% performance of a machine learning technique.

% In this example, we will hold 30% of the data, selected randomly, for
% test phase.
cv = cvpartition(length(bank),'holdout',0.30);

% Training set
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
% Test set
Xtest = X(test(cv),:);
Ytest = Y(test(cv),:);

disp('Training Set')
tabulate(Ytrain)
disp('Test Set')
tabulate(Ytest)



%% Decision Trees
disp('Bank Marketing Campaign - Decision Tree')
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
C_t_train = bsxfun(@rdivide,C_t_train,sum(C_t_train,2)) * 100
C_t_test = bsxfun(@rdivide,C_t_test,sum(C_t_test,2)) * 100

%%Controlling the tree depth

%%Control Depth or "Leafiness"
% % Following code was used to Find the optimal Control Depth. 
% % Uncomment to run and generate the graph
% % Generate minimum leaf occupancies for classification trees from 10 to 1000, spaced exponentially apart:
% 
% leafs = logspace(1,3,10);
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
%%The best leaf size is between about 110 and 130 observations per leaf.

%%DefaultTree
% view(t,'Mode','Graph')
% 
disp('Optimal Tree using Leafiness')
OptimalTree = ClassificationTree.fit(Xtrain,Ytrain,'minleaf',130);
% view(OptimalTree,'mode','graph')

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
C_t_train = bsxfun(@rdivide,C_t_train,sum(C_t_train,2)) * 100
C_t_test = bsxfun(@rdivide,C_t_test,sum(C_t_test,2)) * 100

%%Pruning
%Pruning optimizes tree depth (leafiness) is by merging leaves on the same tree branch
%Unlike in the above section, you do not need to grow a new tree for every node size. 
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
C_t_train = bsxfun(@rdivide,C_t_train,sum(C_t_train,2)) * 100
C_t_test = bsxfun(@rdivide,C_t_test,sum(C_t_test,2)) * 100

%% Prepare Predictors/Response for Neural Networks
% When using neural networks the appropriate way to include categorical
% predictors is as dummy indicator variables. An indicator variable has
% values 0 and 1.
% 
  [XtrainNN, YtrainNN, XtestNN, YtestNN] = preparedataNN_bank(bank, catPred, cv);
 
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
% 
% In order to change Learning Rate, Momentum, Hidden Layer size Check NNfun
disp('Bank Marketing Campaign - Neural Networks')
 [~, net] = NNfun(XtrainNN,YtrainNN);
% 
% Make a prediction for the test set
Y_nn_train = net(XtrainNN');
 Y_nn_train = round(Y_nn_train');

Y_nn_test = net(XtestNN');
Y_nn_test = round(Y_nn_test');

%Compute the confusion matrix
C_nn_train = confusionmat(YtrainNN,Y_nn_train);
C_nn_test = confusionmat(YtestNN,Y_nn_test);
N = sum(C_nn_train(:));
Training_err =( ( N-sum(diag(C_nn_train)) ) / N )* 100
N = sum(C_nn_test(:));
Test_err = (( N-sum(diag(C_nn_test)) ) / N )*100
%Examine the confusion matrix for each class as a percentage of the true class
C_nn_test = bsxfun(@rdivide,C_nn_test,sum(C_nn_test,2)) * 100
C_nn_train = bsxfun(@rdivide,C_nn_train,sum(C_nn_train,2)) * 100


%% Support Vector Machines
% % Support vector machine (SVM) is supported for binary response variables.
% % An SVM classifies data by finding the best hyperplane that separates all
% % data points of one class from those of the other class.
disp('Bank Marketing Campaign - SVM')
opts = statset('MaxIter',30000);
% Train the classifier
tic
svmStruct = svmtrain(Xtrain,Ytrain,'kernel_function','rbf','rbf_sigma',1,'kktviolationlevel',0.1,'options',opts);
%svmStruct = svmtrain(TrainingSet,G1vAll,'kernel_function','polynomial','polyorder',3);
toc

% Make a prediction for the test set
Y_svm_train = svmclassify(svmStruct,Xtrain);
tic
Y_svm_test = svmclassify(svmStruct,Xtest);
toc

C_svm_train = confusionmat(Ytrain,Y_svm_train)
C_svm_test = confusionmat(Ytest,Y_svm_test)
N = sum(C_svm_train(:));
Training_err =( ( N-sum(diag(C_svm_train)) ) / N )* 100
N = sum(C_svm_test(:));
Test_err = (( N-sum(diag(C_svm_test)) ) / N )*100

% Examine the confusion matrix for each class as a percentage of the true class
C_svm_test = bsxfun(@rdivide,C_svm_test,sum(C_svm_test,2)) * 100;
C_svm_train = bsxfun(@rdivide,C_svm_train,sum(C_svm_train,2)) * 100;

 %% Boosting
 
 disp('Bank Marketing Campaign - Boosting')
%Using the same tree
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
title('Bank Marketing Campaign - Boosting')
xlabel('Number of trees');
ylabel('Classification error');
legend('Test','Cross-validation','Training','Location','NE');
Ensemble = fitensemble(Xtrain,Ytrain,'AdaBoostM1',500,'Tree');
Y_b_test  = Ensemble.predict(Xtest);
Y_b_train  = Ensemble.predict(Xtrain);

C_b_train = confusionmat(Ytrain,Y_b_train);
C_b_test = confusionmat(Ytest,Y_b_test);
N = sum(C_b_train(:));
Training_err =( ( N-sum(diag(C_b_train)) ) / N )* 100
N = sum(C_b_test(:));
Test_err = (( N-sum(diag(C_b_test)) ) / N )*100

%Examine the confusion matrix for each class as a percentage of the true class
C_b_test = bsxfun(@rdivide,C_b_test,sum(C_b_test,2)) * 100;
C_b_train = bsxfun(@rdivide,C_b_train,sum(C_b_train,2)) * 100;


%% K Nearest Neighbors

 disp('Bank Marketing Campaign - K Nearest Neighbors')
%For checking the training and testing time
tic
knn = ClassificationKNN.fit(Xtrain,Ytrain,'Distance','seuclidean','DistanceWeight','inverse','NumNeighbors',5);
toc
tic
Y_knn = knn.predict(Xtrain);
toc
% Plotting crossvalidation, training and testing error
%Varied K from 1 to 10,
K = 1:1:10;
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
xlabel('Number of Neighbors');
ylabel('Error');
legend('Cross Validation Error','Training error','Test Error');

%% Shut Down Workers
% Release the workers if there is no more work for them

if matlabpool('size') > 0
    matlabpool close
end
