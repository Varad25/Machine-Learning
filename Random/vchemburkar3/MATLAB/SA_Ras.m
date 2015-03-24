function [x,fval,exitflag,output] = SA_Ras(x0)
%% This is an auto generated MATLAB file from Optimization Tool.

%% Start with the default options
options = saoptimset;
%% Modify options setting
options = saoptimset(options,'Display', 'off');
options = saoptimset(options,'PlotFcns', { @saplotbestf });
[x,fval,exitflag,output] = ...
simulannealbnd(@rastriginsfcn,x0,[],[],options);
