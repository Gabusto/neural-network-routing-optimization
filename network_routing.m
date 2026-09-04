Лістинг Б.1 — Відтворюваний MATLAB-скрипт для варіанта 1
%% Нейронна мережа для визначення мінімального остовного дерева
% Варіант 1, група К933 clear; clc; close all;

%% 1. Топологія мережі та базові затримки
% Кожний рядок edges відповідає ребру t1...t28. edges = [ ...
1 3;  1 9;  2 3; 3 12;  3 4;  3 6; 4 7; ...
4 9;  5 6;  5 7;  6 7; 7 8; 9 10; 10 11; ...
11 12; 12 13; 5 14; 6 14; 13 14; 13 16; 14 16; ...
15 16; 14 15; 16 17; 15 18; 17 18; 18 19; 19 20];


baseDelay = [ ...
4.0 9.0 3.0 11.0 5.0 6.0 8.0 7.0 3.8 4.0 7.0 2.0 ...
4.0 3.0 3.5 2.5 5.0 5.2 7.0 5.0 5.2 3.0 3.2 5.0 ...
4.8 5.1 2.0 2.5]';

nodeCount = 20; edgeCount = size(edges, 1); sampleCount = 100;

%% 2. Формування inputMatrix і targetMatrix rng(933, 'twister');
inputMatrix = zeros(edgeCount, sampleCount); targetMatrix = zeros(edgeCount, sampleCount);

for k = 1:sampleCount
variation = 0.85 + 0.30 * rand(edgeCount, 1); inputMatrix(:, k) = round(baseDelay .* variation, 1); targetMatrix(:, k) = kruskalTarget( ...
edges, inputMatrix(:, k), nodeCount); end
 
assert(all(sum(targetMatrix, 1) == nodeCount - 1), ... 'Кожний цільовий вектор повинен містити 19 одиниць.');

%% 3. Фіксований поділ 70/15/15 rng(27, 'twister');
order = randperm(sampleCount);

trainInd = order(1:70); valInd = order(71:85); testInd = order(86:100);

%% 4. Створення нейронної мережі 28-34-28
hiddenNeuronCount = 34;
net = feedforwardnet(hiddenNeuronCount, 'trainlm');


net.layers{1}.transferFcn = 'tansig'; net.layers{2}.transferFcn = 'purelin'; net.performFcn = 'mse';

net.divideFcn = 'divideind'; net.divideParam.trainInd = trainInd; net.divideParam.valInd = valInd; net.divideParam.testInd = testInd;

net.trainParam.max_fail = 6; net.trainParam.showWindow = true;

%% 5. Навчання та обчислення виходів [net, tr] = train(net, inputMatrix, targetMatrix); outputMatrix = net(inputMatrix); binaryOutput = outputMatrix >= 0.5;

mseTrain = perform(net, targetMatrix(:, trainInd), ... outputMatrix(:, trainInd));
mseVal = perform(net, targetMatrix(:, valInd), ... outputMatrix(:, valInd));
mseTest = perform(net, targetMatrix(:, testInd), ... outputMatrix(:, testInd));
 
fullMatch = all(binaryOutput(:, testInd) == ... targetMatrix(:, testInd), 1); fullMatchAccuracy = mean(fullMatch) * 100;

edgeAccuracy = mean(binaryOutput(:, testInd) == ... targetMatrix(:, testInd), 'all') * 100;

fprintf('MSE train = %.4f\n', mseTrain); fprintf('MSE validation = %.4f\n', mseVal); fprintf('MSE test = %.4f\n', mseTest);
fprintf('Повний збіг тестових векторів = %.2f %%\n', ... fullMatchAccuracy);
fprintf('Поелементна точність = %.2f %%\n', edgeAccuracy);


%% 6. Перевірка структурних обмежень for k = testInd
selected = find(binaryOutput(:, k));
if numel(selected) ~= nodeCount - 1 || ...
~isTree(edges(selected, :), nodeCount) warning('Приклад %d потребує постобробки.', k); end
end

%% 7. Графіки та збереження результатів figure; plotperform(tr);
figure; plotregression(targetMatrix(:, trainInd), ... outputMatrix(:, trainInd), 'Training', ... targetMatrix(:, valInd), outputMatrix(:, valInd), ... 'Validation', targetMatrix(:, testInd), ... outputMatrix(:, testInd), 'Testing');
figure; ploterrhist(targetMatrix - outputMatrix);

save('nn_mst_k933.mat', 'net', 'tr', 'inputMatrix', ... 'targetMatrix', 'outputMatrix', 'binaryOutput', ... 'trainInd', 'valInd', 'testInd');

%% 8. Дослідження кількості нейронів та алгоритму навчання hiddenSet = [10 34 90];
 
trainFunctions = {'trainlm', 'trainbr', 'trainscg'};
experimentMSE = zeros(numel(hiddenSet), numel(trainFunctions)); experimentEpochs = zeros(size(experimentMSE));

for i = 1:numel(hiddenSet)
for j = 1:numel(trainFunctions) rng(1000 + 10*i + j, 'twister');
netExp = feedforwardnet(hiddenSet(i), ... trainFunctions{j}); netExp.layers{1}.transferFcn = 'tansig'; netExp.layers{2}.transferFcn = 'purelin'; netExp.performFcn = 'mse'; netExp.divideFcn = 'divideind'; netExp.divideParam.trainInd = trainInd; netExp.divideParam.valInd = valInd; netExp.divideParam.testInd = testInd; netExp.trainParam.showWindow = false;

[netExp, trExp] = train(netExp, ... inputMatrix, targetMatrix);
yExp = netExp(inputMatrix(:, testInd)); experimentMSE(i, j) = mse( ... targetMatrix(:, testInd) - yExp); experimentEpochs(i, j) = trExp.num_epochs; end
end


disp('MSE для [trainlm, trainbr, trainscg]:'); disp(array2table(experimentMSE, ...
'RowNames', compose('%d_neurons', hiddenSet), ... 'VariableNames', trainFunctions));

disp('Кількість епох:'); disp(array2table(experimentEpochs, ... 'RowNames', compose('%d_neurons', hiddenSet), ... 'VariableNames', trainFunctions));

%% Локальні функції
function target = kruskalTarget(edges, weights, nodeCount)
 
edgeCount = size(edges, 1);
[~, order] = sortrows([weights, (1:edgeCount)'], [1 2]); parent = 1:nodeCount;
rankValue = zeros(1, nodeCount); target = zeros(edgeCount, 1); selectedCount = 0;

for q = 1:edgeCount edgeIndex = order(q);
u = edges(edgeIndex, 1); v = edges(edgeIndex, 2);

[rootU, parent] = findRoot(parent, u); [rootV, parent] = findRoot(parent, v);

if rootU ~= rootV target(edgeIndex) = 1; selectedCount = selectedCount + 1; [parent, rankValue] = unionSets( ... parent, rankValue, rootU, rootV); end

if selectedCount == nodeCount - 1 break;
end end

if selectedCount ~= nodeCount - 1 error('Граф не є зв’язним.');
end end

function result = isTree(selectedEdges, nodeCount) graphObject = graph(selectedEdges(:, 1), ... selectedEdges(:, 2), [], nodeCount);
result = numedges(graphObject) == nodeCount - 1 && ... max(conncomp(graphObject)) == 1;
end
 
function [root, parent] = findRoot(parent, vertex) root = vertex;
while parent(root) ~= root root = parent(root);
end
while parent(vertex) ~= vertex nextVertex = parent(vertex); parent(vertex) = root;
vertex = nextVertex; end
end

function [parent, rankValue] = unionSets( ... parent, rankValue, rootA, rootB)
if rankValue(rootA) < rankValue(rootB) parent(rootA) = rootB;
elseif rankValue(rootA) > rankValue(rootB) parent(rootB) = rootA;
else
parent(rootB) = rootA;
rankValue(rootA) = rankValue(rootA) + 1; end
end
