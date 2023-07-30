%% Synthetic Data Generation by Bi-LSTM Algorithm
% Developed by Seyed Muhammad Hossein Mousavi - July 2023
clear;
clc;
close all;
% Loading the dataset
load fisheriris.mat;
data=reshape(meas,1,[]); % Preprocessing - convert matrix to vector
% The current dataset is iris. If you changed the dataset, you have to change the "Target",
% too as it is the label vector.
Target(1:50)=1;Target(51:100)=2;Target(101:150)=3;Target=Target'; % Original labels
% More the volume value the more complexity and synthetic data . 2 is
% double the original dataset's size. 
Volume=3;
for z=1:Volume
sd=size(data);
numTimeStepsTrain = sd(1,2);
mu = mean(data);
sig = std(data);
dataTrainStandardized = (data - mu) / sig;
XTrain = dataTrainStandardized;
YTrain = dataTrainStandardized;
% Define LSTM Network Architecture
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 250;
layers = [ ...
sequenceInputLayer(numFeatures)
lstmLayer(numHiddenUnits)
fullyConnectedLayer(numResponses)
regressionLayer];
options = trainingOptions('adam', ...
'MaxEpochs',30, ...
'GradientThreshold',1, ...
'InitialLearnRate',0.009, ...
'LearnRateSchedule','piecewise', ...
'LearnRateDropPeriod',256, ...
'LearnRateDropFactor',0.2, ...
'Verbose',0, ...
'Plots','training-progress');
net = trainNetwork(XTrain,YTrain,layers,options);

%% Forecast Future Time Steps
dataTestStandardized = (data - mu) / sig;
XTest = dataTestStandardized;
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));
numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
[net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

% Update Network State with Observed Values
net = resetState(net);
net = predictAndUpdateState(net,XTrain);
YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
[net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end
YPred = sig*YPred + mu;
Synthetic{z}=YPred;
rmse = sqrt(mean((YPred-data).^2))*0.00001;
RMSE(z)=rmse;
end
Synthetic=Synthetic';
% Converting cell to matrix (the last time)
Synthetic2 = cell2mat(Synthetic);
Synthetic2=Synthetic2';
% Converting matrix to cell
P = size(data); P = P (1,2);
S = size(Synthetic{z}); SO = size (meas); SF = SO (1,2); SO = SO (1,1); SS = S (1,2); 
for i = 1 : Volume
Generated1{i}=reshape(Synthetic2(:,i),[SO,SF]);
Generated1{i}(:,end+1)=Target;
end
% Converting cell to matrix (the last time)
Synthetic3 = cell2mat(Generated1');
SyntheticData=Synthetic3(:,1:end-1);
SyntheticLbl=Synthetic3(:,end);

%% Plot data and classes
Feature1=2;
Feature2=3;
f1=meas(:,Feature1); % feature1
f2=meas(:,Feature2); % feature 2
ff1=SyntheticData(:,Feature1); % feature1
ff2=SyntheticData(:,Feature2); % feature 2
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
plot(meas, 'linewidth',1); title('Original Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,2)
plot(SyntheticData, 'linewidth',1); title('Synthetic Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,3)
gscatter(f1,f2,Target,'rkgb','.',20); title('Original');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,4)
gscatter(ff1,ff2,SyntheticLbl,'rkgb','.',20); title('Synthetic');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;

%% Train and Test
% Training Synthetic dataset by SVM
Mdlsvm  = fitcecoc(SyntheticData,SyntheticLbl); CVMdlsvm = crossval(Mdlsvm); 
SVMError = kfoldLoss(CVMdlsvm); SVMAccAugTrain = (1 - SVMError)*100;
% Predict new samples (the whole original dataset)
[label5,score5,cost5] = predict(Mdlsvm,meas);
% Test error and accuracy calculations
sizlbl=size(Target); sizlbl=sizlbl(1,1);
countersvm=0; % Misclassifications places
misindexsvm=0; % Misclassifications indexes
for i=1:sizlbl
if Target(i)~=label5(i)
misindex(i)=i; countersvm=countersvm+1; end; end
% Testing the accuracy
TestErrAugsvm = countersvm*100/sizlbl; SVMAccAugTest = 100 - TestErrAugsvm;
% Result SVM
AugResSVM = [' Synthetic Train SVM "',num2str(SVMAccAugTrain),'" Test on Original Dataset"', num2str(SVMAccAugTest),'"'];
disp(AugResSVM);
