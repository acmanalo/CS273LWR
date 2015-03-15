%% Load Data
kaggleX = load('data/kaggle.X1.train.txt');
kaggleY = load('data/kaggle.Y.train.txt');
kaggleTestData = load('data/kaggle.X1.test.txt');

[xtr, xte, ytr, yte] = splitData(kaggleX, kaggleY, .75);

%% 
clearvars -except kaggleX kaggleY xtr xte ytr yte kaggleTestData
rand('state',0);

% Normalize the data
% [xtr, mu, sigma] = zscore(xtr);
% xte = normalize(xte, mu, sigma);

nn = nnsetup([91 40 1]); % 3 layers, 91 input, variable hidden, 1 output
nn.activation_function = 'sigm';
nn.learningRate =.05;
nn.output = 'linear';
opts.numepochs =  20;   %  Number of full sweeps through data
opts.batchsize = 1000;  %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, xtr, ytr, opts);

% testData = kaggleTestData;
% testData = normalize(testData, mu, sigma);
ypredicted = nnpredict(nn, kaggleTestData);

pred = nnpredict(nn, xte);
unique(pred);
maxKaggle = max(ytr);
minKaggle = min(ytr);
maxPred = max(ypredicted) - min(ypredicted);
minPred = min(ypredicted);

for i = 1:length(ypredicted)
    ypredicted(i) = (ypredicted(i) - minPred) * maxKaggle / maxPred;
end

[er, bad] = nntest(nn, xte, yte);

fh = fopen('predictions.csv','w');  % open file for upload
fprintf(fh,'ID,Prediction\n');      % output header line
for i=1:length(ypredicted),
    fprintf(fh,'%d,%d\n',i,ypredicted(i));  % output each prediction
end;
fclose(fh);                         % close the file