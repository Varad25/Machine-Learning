function err = trainnn (w)

% w has 5*14+5 = 260 weights

% pack weights in the matrices for net
load bank_NN_Data.mat;

net.IW{1} = reshape ( w(1:(5*51)), 5, 51 );
net.LW{2} = reshape ( w((5*51+1):(5*51+5)), 1, 5 );


%calc error
err = ( sum ( (YtrainNN' - sim (net, XtrainNN')).^2 ) + 0.5 * sum(w.*w) ) / (size(XtrainNN,2));

end