%% Random Hill Climbing
tic 
[x,fval,exitflag,output] = RHC_NN([rand(260,1)]);
toc
 %training error
 fval
 outptut
 %testing error
 test_error = performnn(x)
 
%% Genetic Algorithm
 tic
 [x,fval,exitflag,output,population,score] = GA_NN(260);
 toc
 %Objective Function value
 fval
 output
 
 %Testing error
 test_error = performnn(x)
 
 %% Simulated Annealing
 tic
  [x,fval,exitflag,output] = SA_NN([rand(260,1)]);
  toc
  
  %Objective Function value
  fval
  output
  
  %Testing error
  test_error = performnn(x)