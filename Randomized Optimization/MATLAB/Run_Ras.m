%This script runs Random hill climbing, simulated annealing and Genetic algorithms for rastriginsfcn

%% RHC
%%[x,fval,exitflag,output] = RHC_Ras(intialPoint,InitialMeshSize)

%For wide range of intial points
% j = 1;
% for i = -2 : 0.05: 2
% best_points(j,:) = RHC_Ras([i i],0.25);
% j = j+1;
% end

%For particular starting point
tic
best_point_RHC = RHC_Ras([0.5 0.5],0.25);
toc

%% Simulated annealing

tic
best_point_SA =  SA_Ras([0.5 0.5]);
toc

%% Genetic Algorithms
%[x,fval,exitflag,output,population,score] = GA_Ras(nvars,PopulationSize_Data)

tic
best_point = GA_Ras(2,50);
toc
i = 1;
for population = 20:5:500
    [~,fval(i),~] = GA_Ras(2,population);
    i=i+1;
end


figure
plot(20:5:500,fval)
title('fvalue vs population')
%For wide range of intial points
% j = 1;
% for i = -2 : 0.05: 2
% best_points(j,:) = RHC_Ras([i i],0.25);
% j = j+1;
% end