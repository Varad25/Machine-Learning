close all;
clear all;
%% Load Data
load('leafData.mat');
X = leafData(:,3:16);
X = double(X);
labels = double(leafData(:,1));

%% K - Means
rng(1);
k = 25;
tic
[id_k,C] = kmeans(X,k,'Distance','correlation','Display','final','Replicate',300);
toc
figure;
[silh,h] = silhouette(X,id_k,'correlation');
h = gca;
h.Children.EdgeColor = [.8 .8 1];
xlabel 'Silhouette Value';
ylabel 'Cluster';
mean(silh)

 %% EM

k = 10;
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
figure
hist(id_em)

% 
%% PCA
dim = 14;
K=25;
%[COEFF, SCORE, LATENT] = pca(X);
[COEFF, SCORE, LATENT] = pca(X,'NumComponents',dim);
[id_pca,C] = kmeans(SCORE,K,'Distance','correlation','Display','final');
figure;
[silh,h] = silhouette(SCORE,id_pca,'correlation');
h = gca;
h.Children.EdgeColor = [.8 .8 1];
xlabel 'Silhouette Value';
ylabel 'Cluster';
mean(silh)
% hist([id_pca labels])
[id_pca_em,model, llh]= emgm(SCORE',K);
 E = X' - COEFF*(COEFF'*X');
 err = sum(sum(E.*E))

%% ICA
k=30;
 tic
 [z_ic A T mean_z] = myICA(X',6);
 toc
 [id_ica,C] = kmeans(z_ic',k,'Distance','sqeuclidean','Display','final');
 
 [id_ica_em,model, llh]= emgm(z_ic,k);
 
kurtosis(z_ic')
 E = X' - A'*(A*X');
 err = sum(sum(E.*E))

%% RP
 dim = 4;
d = size(X,2);
for i = 1:1000
A = randn( dim, d );  % random iid ~N(0,1)
oA = orth( A.' ).'; % orthogonal rows
z_rp = oA*X';
%[id_rp,C] = kmeans(z_rp',20,'Distance','sqeuclidean');

 E = X' - oA'*z_rp;
 err(i) = sum(sum(E.*E));
 if i == 1
     minr = err(i);
     z_rp_min = z_rp;
    %id_rp_min = id_rp;
 end
 if  minr > err(i) 
     z_rp_min = z_rp;
   % id_rp_min = id_rp;
 end
end

%% Sequential FS


c = cvpartition(labels,'k',10);
opts = statset('display','iter');
fun = @(XT,yT,Xt,yt)...
      (sum(~strcmp(yt,classify(Xt,XT,yT,'linear'))));

[fs,history] = sequentialfs(fun,X,labels,'cv',c,'options',opts);
