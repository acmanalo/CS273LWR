clc; clear; close all;
%%
Xt1=load('kaggle/kaggle.X1.train.txt');     % load the text file
Yt=load('kaggle/kaggle.Y.train.txt');     % load the text file
Xte1=load('kaggle/kaggle.X1.test.txt');     % load the text fil

%%
hold on;
plot(Xt1(:,1), Yt,'g.'); plot(Xt1(:,2),Yt, 'b.'); 

clf

%%
%normalizing
[n,p] = size(Xt1)
% Compute the mean of each column
mu = mean(Xt1); sigma=sqrt(var(Xt1));
% Create a matrix of mean values by
% replicating the mu vector for n rows
MeanMat = repmat(mu,n,1); sigmat=repmat(sigma,n,1);
% Subtract the column mean from each element
% in that column
x_train = (Xt1 - MeanMat)./sigmat;
%%
[n2,p2] = size(Xte1)
% Compute the mean of each column
mu = mean(Xte1); sd=sqrt(var(Xte1));
% Create a matrix of mean values by
% replicating the mu vector for n rows
MeanMat = repmat(mu,n2,1); sdmat=repmat(sd,n2,1);
% Subtract the column mean from each element
% in that column
x_test= (Xte1 - MeanMat)./sdmat;
%%
%Adding seasonal dummies
D1=zeros(length(Xt1),1);
for i=1:length(Xt1),
   if(Xt1(i,1)<100); D1(i)=1;
   end;
end;

D2=zeros(length(Xt1),1);
for i=1:length(Xt1),
   if(Xt1(i,1)>140 && Xt1(i,1)<250 ); D2(i)=1;
   end;
end;

D3=zeros(length(Xt1),1);
for i=1:length(Xt1),
   if(Xt1(i,1)>300); D3(i)=1;
   end;
end;
%%
% seasonal dummies for test
De1=zeros(length(Xte1),1);
for i=1:length(Xte1),
   if(Xte1(i,1)<100); De1(i)=1;
   end;
end;

De2=zeros(length(Xte1),1);
for i=1:length(Xte1),
   if(Xte1(i,1)>140 && Xte1(i,1)<250 ); De2(i)=1;
   end;
end;

De3=zeros(length(Xte1),1);
for i=1:length(Xte1),
   if(Xte1(i,1)>300); De3(i)=1;
   end;
end;
%%
%Adding dummies to Xmatrix
%%

Xtrainmat = [ones(size(Xt1, 1),1) x_train]; Xtestmat= [ones(size(Xte1, 1),1) x_test];

Xtrmat=[D1 D2 Xtrainmat];Xtemat=[De1 De2 Xtestmat];
ntrain = size(Xtrainmat,1);
ntest = size(Xtestmat,1);

%%
%%%%Removing outliers
for i=1:length(
[Xt1out, IDX, OUTLIERS] = deleteoutliers(Xt1);
 %{ 
delete outliers function:
[B, IDX, OUTLIERS] = DELETEOUTLIERS(A, ALPHA, REP)

For input vector A, returns a vector B with outliers (at the significance level alpha) removed. 
Also, optional output argument idx returns the indices in A of outlier values. 
Optional output argument outliers returns the outlying values in A.
 
ALPHA is the significance level for determination of outliers. If not provided, alpha defaults to 0.05.
  
REP is an optional argument that forces the replacement of removed elements with NaNs to preserve the length of a.
%}



%%
%{
for j=1:ntest,
     for i=1:ntrain,
            covx=cov(Xtrmat(i,2),Xtemat(j,2));
            W(i,i) = 0.5 * (exp(-((Xtrmat(i,2)-Xtemat(j,2))^2))-exp(covx));
            i
     end;
    theta = inv(Xtrmat'*W*Xtrmat+eps*eye(size(Xt1)))*Xtrmat'*W*Yt;
    LWRYp(j) = theta*Xtrmat(j);
    j
end;
%}
%%

for j=1:ntrain,
    covx=cov(Xtrmat(i,:),Xtemat(i,:);
    W(i,i) = 0.5 * (exp(-(repmat(Xtrmat(j,:),94)-Xtemat)^2))-exp(covx));
  
    theta = inv(Xtrmat'*W*Xtrmat+eps*eye(size(Xt1)))*Xtrmat'*W*Yt;
    LWRYp(j) = theta*Xtrmat(j);
    j
end;

%%
figure(1); hold on;
plot (LWRYp);
plot(Yt(11:15))

%%
ç


%Creating new matrix with dummies
Xtraindum=[D1 D2 D3 Xt1out];
