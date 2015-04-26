addpath('./MDPtoolbox');
% change grid here
% grid_easy, grid_easy_10, grid_easy_20
my_grid = grid_easy;
figure;
colormap(gray); imagesc(my_grid);
set(gca,'XTick',[0.5:1:31.5],...
        'YTick',[0.5:1:31.5])
grid on;
title ('Grid')

%prepare
sz = size(my_grid,1)-2;
% p =0.9
[P,R] = grid_to_MDP (my_grid,0.9);


%solve
%Policy Iteration
[V, policy, iter, cpu_time] = mdp_policy_iteration ( P, R, 0.9 );

% Value Iteration
%[V, policy, iter, cpu_time] = mdp_value_iteration ( P, R, 0.9 );

%display ( [ num2str(iter) '  &  ' num2str(cpu_time) ] );
% % Q-Learning
% tic
% [Q, V, policy, mean_discrepancy] = mdp_Q_learning(P, R, 0.9,20000000);
% toc

%plot policy
figure;
colormap(gray); imagesc(my_grid);
hold on

dir_x = zeros(sz,sz);
dir_y = zeros(sz,sz);
len = 0.5;

for i=1:sz
    for j=1:sz
        
        dir = policy( (i-1)*sz+j );
        
        if ( my_grid(i+1,j+1) == -1 || my_grid(i+1,j+1) == 1 )
            continue;
        end
        
        if ( dir == 1 )
            dir_x(i,j) = -len;
        elseif (dir ==2 )
            dir_x(i,j) = len;
        elseif ( dir == 3)
            dir_y(i,j) = -len;
        else
            dir_y (i,j) = len;
        end
    end
end

[ x y ] = meshgrid (2:sz+1,2:sz+1);
quiver (x,y,dir_x,dir_y );

set(gca,'XTick',[0.5:1:31.5],...
        'YTick',[0.5:1:31.5])
grid on;
title ('Grid - solution');
hold off;

%plot V-values
figure;colormap(gray); imagesc(my_grid);
set(gca,'XTick',[0.5:1:15.5],...
        'YTick',[0.5:1:15.5])
grid on;
title ('grid - V values')

for i=1:sz
    for j=1:sz
        text( j+1-0.2, i+1, num2str(V( (i-1)*sz+j ), '%10.2f' ), 'BackgroundColor',[1 1 1] );
    end
end

figure;
colormap(gray);
imagesc( reshape(V,sz,sz)' );
title ('V values');

