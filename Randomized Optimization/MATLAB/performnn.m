function res = performnn (w)

load bank_NN_Data.mat;

%Change the weights
net.IW{1} = reshape ( w(1:(5*51)), 5, 51 );
net.LW{2} = reshape ( w((5*51+1):(5*51+5)), 1, 5 );


Y_nn_test = net(XtestNN');
 Y_nn_test = round(Y_nn_test');
C_nn_test = confusionmat(YtestNN,Y_nn_test);
N = sum(C_nn_test(:));
res = (( N-sum(diag(C_nn_test)) ) / N )*100;


end