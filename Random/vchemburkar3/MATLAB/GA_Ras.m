function [x,fval,exitflag,output,population,score] = GA_Ras(nvars,PopulationSize_Data)
%% This is an auto generated MATLAB file from Optimization Tool.

%% Start with the default options
options = gaoptimset;
%% Modify options setting
options = gaoptimset(options,'PopulationSize', PopulationSize_Data);
options = gaoptimset(options,'Display', 'off');
%options = gaoptimset(options,'PlotFcns', { @gaplotbestf });
[x,fval,exitflag,output,population,score] = ...
ga(@rastriginsfcn,nvars,[],[],[],[],[],[],[],[],options);
