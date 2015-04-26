function [x,fval,exitflag,output] = SA_NN(x0)
%% This is an auto generated MATLAB file from Optimization Tool.

%% Start with the default options
options = saoptimset;
%% Modify options setting
options = saoptimset(options,'Display', 'off','timelimit',400);
[x,fval,exitflag,output] = ...
simulannealbnd(@trainnn,x0,[],[],options);
