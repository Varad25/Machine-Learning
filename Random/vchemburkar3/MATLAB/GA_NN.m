function [x,fval,exitflag,output,population,score] = GA_NN(nvars)
%% This is an auto generated MATLAB file from Optimization Tool.

%% Start with the default options
options = gaoptimset;
%% Modify options setting
options = gaoptimset(options,'Display', 'off');
[x,fval,exitflag,output,population,score] = ...
ga(@trainnn,nvars,[],[],[],[],[],[],[],[],options);
