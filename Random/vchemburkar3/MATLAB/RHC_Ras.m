function [x,fval,exitflag,output] = RHC_Ras(x0,InitialMeshSize_Data)
%% This is an auto generated MATLAB file from Optimization Tool.

%% Start with the default options
options = psoptimset;
%% Modify options setting
options = psoptimset(options,'InitialMeshSize', InitialMeshSize_Data);
options = psoptimset(options,'CompleteSearch', 'off');
options = psoptimset(options,'Display', 'off');
options = psoptimset(options,'PlotFcns', {  @psplotbestf @psplotfuncount });
[x,fval,exitflag,output] = ...
patternsearch(@rastriginsfcn,x0,[],[],[],[],[],[],[],options);
