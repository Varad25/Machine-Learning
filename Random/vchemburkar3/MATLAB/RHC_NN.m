function [x,fval,exitflag,output] = RHC_NN(x0)
%% This is an auto generated MATLAB file from Optimization Tool.

%% Start with the default options
options = psoptimset;
%% Modify options setting
options = psoptimset(options,'Display', 'off','timelimit',400);
options = psoptimset(options,'PlotFcns', { @psplotbestf });
[x,fval,exitflag,output] = ...
patternsearch(@trainnn,x0,[],[],[],[],[],[],[],options);
