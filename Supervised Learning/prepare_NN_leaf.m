function [ XtrainNN, YtrainNN, XtestNN, YtestNN ] = prepare_NN_leaf( leaf, cv)


Y = leaf.Class;
disp('Leaf Classification')
%tabulate(Y)
% Predictor matrix
X = double(leaf(:,3:end));

YNN = zeros(size(Y),30);
for i = 1:size(Y)
    YNN(i,Y(i))= 1;
end


% Use the same partition for cross validation
% Training set
XtrainNN = X(training(cv),:);
YtrainNN = YNN(training(cv),:);
% Test set
XtestNN = X(test(cv),:);
YtestNN = YNN(test(cv),:);
