%% Load data
kaggleX = load('data/kaggle.X1.train.txt');
kaggleY = load('data/kaggle.Y.train.txt');
kaggleTestData = load('data/kaggle.X1.test.txt');
normKaggle = normalizeData(kaggleX);
normTestData = normalizeData(kaggleTestData);
%% Test on smaller set
rand('state',0)

nTruncated = 60000; % Number of data from the truncated set
indices = randi(length(normKaggle), nTruncated, 1);
truncatedX = normKaggle(indices, :);
truncatedY = kaggleY(indices, :);

[xtr, xte, ytr, yte] = splitData(truncatedX, truncatedY, .75);

%%
nDivisions = 19;

MSE = zeros(nDivisions, 1);
c = linspace(.05, 1, nDivisions);
for k = 1:nDivisions
    predictedKaggle = zeros(size(yte));
    for i = 1:size(xte, 1)
        predictedKaggle(i) = lwrPredict(xtr, ytr, xte(i, :), c(k));
        predictedKaggle(i)
    end
    MSE(k) = mse(predictedKaggle, yte)
    k
end

%% Cross-validate on full training set
[xtr, xte, ytr, yte] = splitData(normKaggle, kaggleY, .75);

for i = 1:size(xte, 1)
    lwrPredict(xtr, ytr, xte(i, :), .25)
end


%% Test on Full Set
predictedKaggle = zeros(size(kaggleTestData, 1), 1);
done = 0;
parfor i = 1:length(kaggleTestData)
    i
    done = done+1
    predictedKaggle(i) = lwrPredict(normKaggle, kaggleY, normTestData(i,:), .25);
end

fh = fopen('predictions.csv','w');  % open file for upload
fprintf(fh,'ID,Prediction\n');      % output header line
for i=1:length(predictedKaggle),
    fprintf(fh,'%d,%d\n',i,predictedKaggle(i));  % output each prediction
end;
fclose(fh);                         % close the file

%% Run
clearvars -except kaggleX kaggleY xtr xte ytr yte kaggleTestData iris X Y
rand('state',0)

[xtr2, xte2, ytr2, yte2] = splitData(xtr, ytr, .66);

[xtr2, mu, sigma] = zscore(xtr2);
xte2 = normalize(xte2, mu, sigma);

nPoints = size(xte2, 1);

predictions = zeros(nPoints, 1);
for i = 1:nPoints
    predictions(i) = lwrPredict(xtr2, ytr2, xte2(i, :), 1);
end

%% Linear Regress from class
clearvars -except kaggleX kaggleY xtr xte ytr yte kaggleTestData iris X Y
rand('state',0)

[xtr2, xte2, ytr2, yte2] = splitData(xtr(:, 2:end), ytr, .66);

[xtr2, mu, sigma] = zscore(xtr2);
xte2 = normalize(xte2, mu, sigma);

lr = linearRegress(xtr2, ytr2);
lr = train(lr, xtr2, ytr2, 0);

mse(lr, xte2, yte2)

%% Test Regression
fid = fopen('Data/machine.data.txt');
data = textscan(fid, '%s%s%d%d%d%d%d%d%d%d', 207, 'delimiter', ',');
compData = zeros(207, 8);

for i = 3:10
    compData(:, i-2) = data{i};
end
X = compData(:, 1:7); Y = compData(:, 8);

XScaled = zeros(size(X));
for i = 1:length(X)
    XScaled(i, :) = (X(i, :) - min(X)) / (max(X) - min(X));
end

[xtr, xte, ytr, yte] = splitData(XScaled, Y, .75);

% lr = linearRegress(xtr, ytr);
% lr = train(lr, xtr, ytr, .05);
% predictions = predict(lr, xte);
% mse(lr, xte, yte)

lwrPredictions = zeros(length(xte), 1);
for i = 1:length(xte)
    lwrPredictions(i) = lwrPredict(xtr, ytr, xte(i, :), .25);
end

x = linspace(1, length(xte), length(xte));
fig()
hold on
scatter(x, yte, 'filled');
scatter(x, lwrPredictions, 'filled');
mse(yte, lwrPredictions)
legend('Actual', 'Predicted')
