%% Load data
kaggleX = load('Data/kaggle.X1.train.txt');
kaggleY = load('Data/kaggle.Y.train.txt');
kaggleTestData = load('Data/kaggle.X1.test.txt');

%% Normalize and split data
rand('state', 0);

[xNormalized, mu, sigma] = zscore(kaggleX);
test_x = normalize(kaggleTestData, mu, sigma);

[nTr, nInputs] = size(xNormalized);
[xtr, xte, ytr, yte] = splitData(xNormalized, kaggleY, .75);

%% Sample

nTe = size(kaggleTestData, 1);
H1 = 5;
H2 = 10;

nn = nnsetup([nInputs H1 H2 1]);
nn.output = 'linear';
nn.learningRate = .05;
opts = [];
opts.numepochs = 200;
opts.batchsize = 1000;

[nn, L] = nntrain(nn, xtr, ytr, opts, xte, yte);
tmp = nnff(nn, test_x, zeros(nTe, 1));
yHat = tmp.a{end};

fh = fopen('predictions.csv','w');  % open file for upload
fprintf(fh,'ID,Prediction\n');      % output header line
for i=1:length(yHat),
    fprintf(fh,'%d,%d\n',i,yHat(i));  % output each prediction
end;
fclose(fh);                         % close the file

%% Test single hidden layer
% Test training accuracy of a single layer NN
% Find out what number of layers provides a good prediction on our data
nModels = 10;

singleLayerNNs = cell(nModels, 1);

for i = 1:10
    nn = nnsetup([nInputs 5 + 20*i 1]);
    nn.output = 'linear';
    nn.learningRate = .05;
    opts = [];
    opts.numepochs = 100;
    opts.batchsize = 1000;
    
    [nn, L] = nntrain(nn, xtr, ytr, opts, xte, yte);
    singleLayerNNs{i} = nn;
end

%% Test two hidden layers
% Test effects of multiple layers
% Find out what combination of first layer nodes and second layer nodes
% provide the best prediction on our data
nFirstLayers = 10;
nSecondLayers = 10;

multiLayerNNs = cell(nFirstLayers, nSecondLayers);

for i = 1:nFirstLayers
    for j = 1:nSecondLayers
        nn = nnsetup([nInputs 10*i 10*j 1]);
        nn.output = 'linear';
        nn.learningRate = .05;
        opts = [];
        opts.numepochs = 100;
        opts.batchsize = 1000;
        
        [nn, L] = nntrain(nn, xtr, ytr, opts, xte, yte);
        multiLayerNNs{i, j} = nn;
    end
end

%% Test three hidden layers
% Test training accuracy of three hidden layers
% Find out what combination of three layers NNs provide best predictions
nFirstLayers = 4;
nSecondLayers = 4;
nThirdLayers = 4;

threeLayerNNs = cell(nFirstLayers, 1);
for i = 1:nFirstLayers
    temp = cell(nSecondLayers, nThirdLayers);
    for j = 1:nSecondLayers
        for k = 1:nThirdLayers
            nn = nnsetup([nInputs 5 + 25*(j - 1) 5 + 25 * (k-1) 5 + 25*(i-1) 1]);
            nn.output = 'linear';
            nn.learningRate = .05;
            opts = [];
            opts.numepochs = 100;
            opts.batchsize = 1000;
            
            [nn, L] = nntrain(nn, xtr, ytr, opts, xte, yte);
            temp{j, k} = nn;
        end
    end
    threeLayerNNs{i} = temp;
end


%% Evaluate each of the models created
singleLayerMSEs = zeros(1, length(singleLayerNNs));
for i = 1:length(singleLayerNNs)
    temp = nnff(singleLayerNNs{i}, xte, zeros(size(xte, 1), 1));
    yHat = temp.a{end};
    
    singleLayerMSEs(i) = mse(yHat, yte);
end

multiLayerMSEs = zeros(size(multiLayerNNs));
% First dim varies the first layer
% Second dim varies the second layer
for i = 1:size(multiLayerNNs, 1)
    for j = 1:size(multiLayerNNs, 2)
        temp = nnff(multiLayerNNs{i, j}, xte, zeros(size(xte, 1), 1));
        yHat = temp.a{end};
        
        multiLayerMSEs(i, j) = mse(yHat, yte);
    end
end

threeLayerMSEs = zeros(nFirstLayers, nSecondLayers, nThirdLayers);

for i = 1:nFirstLayers
    tempLayer = threeLayerNNs{i};
    for j = 1:nSecondLayers
        for k = 1:nThirdLayers
            temp = nnff(tempLayer{j, k}, xte, zeros(size(xte, 1), 1));
            yHat = temp.a{end};
            
            threeLayerMSEs(i, j, k) = mse(yHat, yte);
        end
    end
end

%% Plot Results
% Single Layer MSE plot
xSingleLayer = [25:20:205];
plot(xSingleLayer, singleLayerMSEs)

% Two Layer MSE Plots
cc = hsv(size(multiLayerMSEs, 1));

fig()
hold on
for i = 1:size(multiLayerMSEs, 1)
    plot([10:10:100], multiLayerMSEs(i, :), 'color', cc(i,:),...
        'DisplayName', ['Layer1 Nodes = ' num2str(i * 10)])
end
legend(gca, 'show')

% Three Layer MSE Plots
cc = hsv(4);

fig()
nodes = [5 30 55 80];
for i = 1:4
    subplot(2,2,i)
    hold on
    
    for j = 1:4
        plot(nodes, reshape(threeLayerMSEs(i, j, :), [4, 1]), 'color', cc(j, :),...
            'DisplayName', ['Layer2 Nodes = ' num2str(nodes(j))]);
    end
    title(['Layer1 Nodes = ' num2str(nodes(i))])
    legend(gca, 'show')
end

%% Examining varying hidden node sizes
hiddenNodesSizes = [10, 25, 50, 100, 150, 300, 500];

nModels = length(hiddenNodesSizes);

varyingHiddenNodes1LayerMSEs = zeros(nModels, 1);

for i = 1:nModels
    nn = nnsetup([nInputs hiddenNodesSizes(i) 1]);
    nn.output = 'linear';
    nn.learningRate = .01;
    opts = [];
    opts.numepochs = 100;
    opts.batchsize = 1000;
    
    [nn, L] = nntrain(nn, xtr, ytr, opts, xte, yte);
    temp = nnff(nn, xte, zeros(size(xte, 1), 1));
    yHat = temp.a{end};
    
    varyingHiddenNodes1LayerMSEs(i) = mse(yHat, yte);
    fprintf('Model: %d\n\n', i)
end

varyingHiddenNodes2LayerMSEs = zeros(nModels, nModels);
for i = 1:nModels
    for j = 1:nModels
        nn = nnsetup([nInputs hiddenNodesSizes(i) hiddenNodesSizes(j) 1]);
        nn.output = 'linear';
        nn.learningRate = .05;
        opts = [];
        opts.numepochs = 100;
        opts.batchsize = 1000;
        
        [nn, L] = nntrain(nn, xtr, ytr, opts, xte, yte);
        temp = nnff(nn, xte, zeros(size(xte, 1), 1));
        yHat = temp.a{end};
        
        varyingHiddenNodes2LayerMSEs(i, j) = mse(yHat, yte);
        fprintf('Model: %d %d\n\n', i, j);
    end
end

%% Examining varying batch sizes
batchSizes = [100, 300, 500, 1000, 1500, 3000, 5000];

nModels = length(batchSizes);

varyingBatchSizes1LayerMSEs = zeros(nModels, 1);

for i = 1:nModels
    nn = nnsetup([nInputs 50 1]);
    nn.output = 'linear';
    nn.learningRate = .05;
    opts = [];
    opts.numepochs = 100;
    opts.batchsize = batchSizes(i);
    
    [nn, L] = nntrain(nn, xtr, ytr, opts, xte, yte);
    temp = nnff(nn, xte, zeros(size(xte, 1), 1));
    yHat = temp.a{end};
    
    varyingBatchSizes1LayerMSEs(i) = mse(yHat, yte);
    
    sprintf('Model: %d', i)
end

varyingBatchSizes2LayerMSEs = zeros(nModels, 1);
for i = 1:nModels
    nn = nnsetup([nInputs 50 50 1]);
    nn.output = 'linear';
    nn.learningRate = .05;
    opts = [];
    opts.numepochs = 100;
    opts.batchsize = batchSizes(i);
    
    [nn, L] = nntrain(nn, xtr, ytr, opts, xte, yte);
    temp = nnff(nn, xte, zeros(size(xte, 1), 1));
    yHat = temp.a{end};
    
    varyingBatchSizes2LayerMSEs(i) = mse(yHat, yte);
    sprintf('Model: %d', i)
end

%% Varying the learning rate
learningRates = [.01 .02 .04 .08 .16];

nLearningRates = length(learningRates);
varyingLearningRates1LayerMSEs = zeros(nLearningRates, 1);

for i = 1:nLearningRates
    nn = nnsetup([nInputs 50 1]);
    nn.output = 'linear';
    nn.learningRate = learningRates(i);
    opts = [];
    opts.numepochs = 100;
    opts.batchsize = 1000;
    
    [nn, L] = nntrain(nn, xtr, ytr, opts, xte, yte);
    temp = nnff(nn, xte, zeros(size(xte, 1), 1));
    yHat = temp.a{end};
    
    varyingLearningRates1LayerMSEs(i) = mse(yHat, yte);
    sprintf('Model: %d', i)
end

varyingLearningRates2LayerMSEs = zeros(nLearningRates, 1);
for i = 1:nLearningRates
    nn = nnsetup([nInputs 50 50 1]);
    nn.output = 'linear';
    nn.learningRate = learningRates(i);
    opts = [];
    opts.numepochs = 100;
    opts.batchsize = 1000;
    
    [nn, L] = nntrain(nn, xtr, ytr, opts, xte, yte);
    temp = nnff(nn, xte, zeros(size(xte, 1), 1));
    yHat = temp.a{end};
    
    varyingLearningRates2LayerMSEs(i) = mse(yHat, yte);
    sprintf('Model: %d', i)
end

%% Plot varying learning rate results
% Learning rate of .16 resulted in diverging MSE
fig()
plot(learningRates(1:4), varyingLearningRates1LayerMSEs(1:4), learningRates(1:4), ...
    varyingLearningRates2LayerMSEs(1:4));
title('Effects of Learning Rates on MSE')
ylabel('MSE')
xlabel('Learning Rate')
legend('1-Layer NN', '2-Layer NN')

%% Plot varying batch size results
fig()
plot(batchSizes, varyingBatchSizes1LayerMSEs, batchSizes, ...
    varyingBatchSizes2LayerMSEs)
legend('1-Layer NN', '2-Layer NN')
ylabel('MSE')
xlabel('Batch size')
title('Effects of Batch Size on MSE')

%% Plot Varying Hidden Node Sizes
fig()
plot(hiddenNodesSizes, varyingHiddenNodes1LayerMSEs)

cc = hsv(length(hiddenNodesSizes));

fig()
hold on
for i = 1:length(hiddenNodesSizes)
    plot(hiddenNodesSizes, varyingHiddenNodes2LayerMSEs(i, :), ...
        'color', cc(i,:), 'DisplayName', ['Layer1 Nodes = ' ...
        num2str(hiddenNodesSizes(i))])
end
legend(gca, 'show')

%% Ensembles of Neural Networks
% Boostrapping
[N, D] = size(xtr);

Nbag = 25;          % Number of learners
Nuse = N;           % Number of data from original set to use

learners = cell(1, Nbag);

% Create all the learners
for i = 1:Nbag
    randIndices = randi(Nuse, Nuse, 1);
    xTrBagged = xtr(randIndices);
    yTrBagged = ytr(randIndices);
    
    nn = nnsetup([nInputs 200 1]);
    nn.output = 'linear';
    nn.learningRate = .02;
    opts = [];
    opts.numepochs = 400;
    opts.batchsize = 500;
    
    [nn, L] = nntrain(nn, xTrBagged, yTrBagged, opts, xte, yte);
    learners{i} = nn;
end











