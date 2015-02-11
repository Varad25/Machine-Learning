function [result] = multisvm(TrainingSet,GroupTrain,TestSet)
%Models a given training set with a corresponding group vector and 
%classifies a given test set using an SVM classifier according to a 
%one vs. all relation. 
%


u=unique(GroupTrain);
numClasses=length(u);
result = zeros(length(TestSet(:,1)),1);
tic
%build models
for k=1:numClasses
    %Vectorized statement that binarizes Group
    %where 1 is the current class and 0 is all other classes
    G1vAll=(GroupTrain==u(k));
    models(k) = svmtrain(TrainingSet,G1vAll,'kernel_function','rbf','rbf_sigma',1);
    %models(k) = svmtrain(TrainingSet,G1vAll,'kernel_function','polynomial','polyorder',1);
end
toc
tic
%classify test cases
for j=1:size(TestSet,1)
    for k=1:numClasses
        if(svmclassify(models(k),TestSet(j,:))) 
            break;
        end
    end
    result(j) = k;
end
toc