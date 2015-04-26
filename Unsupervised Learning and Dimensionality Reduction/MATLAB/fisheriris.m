close all;
clear all;
%% Loading Data and Plotting it
load fisheriris_data.mat;
% figure
% spread(X',ones(1,150));
% title('Original Data');
% xlabel 'Sepal Lengths (cm)';
% ylabel 'Sepal Widths (cm)';
% zlabel 'Petal Lengths (cm)';
% 
% figure
% spread(X', labels);
% title('True Clusters');
% xlabel 'Sepal Lengths (cm)';
% ylabel 'Sepal Widths (cm)';
% zlabel 'Petal Lengths (cm)';
% legend('setosa','versicolor','virginica')

%% K- Means Clustering
k = 3;
rng(1);
tic
[id_k,C] = kmeans(X,k,'Distance','correlation','Display','final');
toc
figure;
[silh,h] = silhouette(X,id_k,'correlation');
h = gca;
h.Children.EdgeColor = [.8 .8 1];
xlabel 'Silhouette Value';
ylabel 'Cluster';
mean(silh)
figure
spread(X',id_k);
str = sprintf('K-means with k = %i Clusters and Correlation distance',k);
title(str);
xlabel 'Sepal Lengths (cm)';
ylabel 'Sepal Widths (cm)';
zlabel 'Petal Lengths (cm)';

%% EM
k = 2;
tic
[id_em,model, llh]= emgm(X',k);
toc
figure;
[silh,h] = silhouette(X,id_em,'correlation');
h = gca;
h.Children.EdgeColor = [.8 .8 1];
xlabel 'Silhouette Value';
ylabel 'Cluster';
mean(silh)
% figure
% spread(X',id_em);
% str = sprintf('K-means with k = %i Clusters',k);
% title(str);
% xlabel 'Sepal Lengths (cm)';
% ylabel 'Sepal Widths (cm)';
% zlabel 'Petal Lengths (cm)';

%% PCA

[COEFF, SCORE, LATENT] = pca(X);
tic
 [COEFF, SCORE, LATENT] = pca(X,'NumComponents',2);
toc
 figure
 spread(SCORE',ones(1,150))
 title('Reduced Data')
  [id_pca,C] = kmeans(SCORE,2,'Distance','correlation','Display','final');
  figure
  spread(SCORE',id_pca);
  title('PCA and k-Means')
  [id_pca_em,model, llh]= emgm(SCORE',2);
  figure
  spread(SCORE',id_pca_em);
  title('PCA and EM')
  %Reconstruction Error
  E = X' - COEFF*(COEFF'*X');
  err = sum(sum(E.*E))
 %% ICA
%  
 k=2;
 tic
 [z_ic A T mean_z] = myICA(X',2);
 toc
 figure
 spread(z_ic,ones(1,150))
 [id_ica,C] = kmeans(z_ic',k,'Distance','sqeuclidean','Display','final');
 figure
 spread(z_ic,id_ica);
  title('ICA and k-Means')
 [id_ica_em,model, llh]= emgm(z_ic,k);
 figure
 spread(z_ic,id_ica_em);
 title('ICA and EM')
 kurtosis(z_ic')
 E = X' - A'*(A*X');
 err = sum(sum(E.*E))

 
%%  Randomized Projection
 k = 2;
d = size(X,2);
for i = 1:1000
A = randn( k, d );  % random iid ~N(0,1)
oA = orth( A.' ).'; % orthogonal rows
z_rp = oA*X';
 %[id_rp,C] = kmeans(z_rp',k,'Distance','sqeuclidean');
 [id_rp, ~]= emgm(z_rp,2);

 E = X' - oA'*z_rp;
 err(i) = sum(sum(E.*E));
 if i == 1
     minr = err(i);
     z_rp_min = z_rp;
     id_rp_min = id_rp;
 end
 if  minr > err(i) 
     z_rp_min = z_rp;
     id_rp_min = id_rp;
 end
end

 figure
 spread(z_rp_min,ones(1,150));
 figure
spread(z_rp_min,id_rp_min);

%% Sequential FS

load fisheriris;
X = randn(150,10);
X(:,[1 3 7 ])= meas(:,[1 2 3]);
y = species;

c = cvpartition(y,'k',10);
opts = statset('display','iter');
fun = @(XT,yT,Xt,yt)...
      (sum(~strcmp(yt,classify(Xt,XT,yT,'quadratic'))));

[fs,history] = sequentialfs(fun,X,y,'cv',c,'options',opts)
 [id_fs,C] = kmeans(X(:,fs),2,'Distance','correlation','Display','final');
 spread(X(:,fs)',id_fs);


