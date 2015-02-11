README

All the algorithms were implemented in MATLAB. 

1.	Files

Bank_Marketing.m - All the algorithms for Bank Marketing are implemented here
LeafClassification.m - All the algorithms for Leaf are implemented
NNfun.m - Above two scripts call this function to run Neural Network
prepare_NN_leaf.m - This script prepares the Leaf Data for Neural Network
preparedataNN_bank.m - This script prepares the Bank Data for Neural Network
multisvm.m - This runs the Multi SVM required for Leaf Data
ImportBankData.m - This imports Bank Data from csv file
importLeafData.m - This imports Leaf Data from csv file
Removequotes.m - Removes unwanted quotes from the Bank Data
bank-full.csv - Bank Data
leaf.csv - Leaf Data
Bank_Attributes.pdf – Information of Bank Marketing Dataset and its attributes
Leaf_Attributes.pdf - Information of Leaf Dataset and its attributes

2.	Running Algorithms with Default Values

Copy all the files under MATLAB's Current folder
To run all the algorithms on Bank Marketing dataset with default values, run Bank_Marketing.m
To run all the algorithms on Leaf dataset with default values, run LeafClassification.m

3.	Modifying the parameters

Most of the algorithms require some modifications to generate all the results documented in the analysis and these modifications are commented out in the scripts. 
Simply uncomment them to run. 
For example, in order to change the kernel of SVM for Bank Marketing, go to SVM section of Bank_Marketing.m and simply uncomment the 284th line and comment the 283rd. 
This changes the kernel from rbf to polynomial. Change the polynomial order, by changing the number after ‘polyorder’. Similarly kernel can be changed for Leaf. 
For Leaf go to multisvm.m and change the kernel. 
Distance metrics for K-NN can be changed and are mentioned in the K-NN sections of both Bank Marketing and Leaf Classification
For Neural Networks, number of Hidden Layers, Size of the layers, learning rate and momentum can be changed in NNfun.m
