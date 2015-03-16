%% Load data
kaggleX = load('Data/kaggle.X1.train.txt');
kaggleY = load('Data/kaggle.Y.train.txt');
kaggleTestData = load('Data/kaggle.X1.test.txt');


%%
[train_x, mu, sigma] = zscore(kaggleX);
test_x = normalize(kaggleTestData, mu, sigma);

rand('state', 0);

nTe = size(kaggleTestData, 1);
[nTr, nInputs] = size(train_x);
H = 5;

nn = nnsetup([nInputs H 1]);
nn.output = 'linear';
nn.learningRate = .05;
opts = [];
opts.numepochs = 500;
opts.batchsize = 1000;

[xtr, xte, ytr, yte] = splitData(train_x, kaggleY, .75);
[nn, L] = nntrain(nn, xtr, ytr, opts, xte, yte);
tmp = nnff(nn, test_x, zeros(nTe, 1));
yHat = tmp.a{end};

fh = fopen('predictions.csv','w');  % open file for upload
fprintf(fh,'ID,Prediction\n');      % output header line
for i=1:length(yHat),
    fprintf(fh,'%d,%d\n',i,yHat(i));  % output each prediction
end;
fclose(fh);                         % close the file

