clear; clc; close all;
%% Part I : only csv data
% Reading data

info = importdata("ADNI_featur%data = readtable('ADNI_features.csv')es.csv")
raw_data=info.data;
data0=raw_data;

data0=data0(all(~isnan(data0),2),:); %delet NaN rows
means=mean(data0); %mean after delete

[m,n]=find(isnan(raw_data(:,2))==1); %find row m in Nan Raw data
for i=1:length(m)
    raw_data(m(i),2:118)=means(2:118); %replace with mean
end


%% Dividing test and train
y=raw_data(:,1);
x=raw_data(:,2:122);
tmpVec = randperm(size(y,1));
M = 73; %60% data as train
x_train = x(tmpVec(1:M),:);
y_train = y(tmpVec(1:M));
x_test = x(tmpVec(M+1:end),:);
y_test = y(tmpVec(M+1:end));


%% Compute features using PCA

fprintf('Running PCA...\n');
[evectors_train, score_train, evalues_train] = pca(x_train');
[evectors_test, score_test, evalues_test] = pca(x_test');

num_eigenvalues = 73; 
evectors_train = evectors_train(:, 1:num_eigenvalues);
evectors_test = evectors_test(:, 1:num_eigenvalues);

mean_x_train = mean(x_train, 2);
shifted_x_train = x_train - repmat(mean_x_train, 1, size(x_train,2));
mean_x_test = mean(x_test, 2);
shifted_x_test = x_test - repmat(mean_x_test, 1, size(x_test,2));

feature_train = evectors_train' * shifted_x_train;
feature_test = evectors_test' * shifted_x_test;
%% Decision tree nodes plot
X = x_train;
Y = y_train;
rng(1); % For reproducibility
MdlDefault = fitctree(X,Y,'CrossVal','on');
numBranches = @(x)sum(x.IsBranch);
mdlDefaultNumSplits = cellfun(numBranches, MdlDefault.Trained);

figure;
histogram(mdlDefaultNumSplits)
view(MdlDefault.Trained{1},'Mode','graph')
%% The average number of splits is around 50

Mdl15 = fitctree(x_train, y_train,'MaxNumSplits',50,'CrossVal','on');
view(Mdl15.Trained{1},'Mode','graph')
classErrorDefault = kfoldLoss(MdlDefault)
classError15 = kfoldLoss(Mdl15)
%Mdl15 is much less complex and performs better than MdlDefault.
%% Decision tree

model = fitcensemble(x_train, y_train);
[~,y_dt_prob] = predict(model, x_test);
[~,y_dt_pred] = max(y_dt_prob');

% Post processing k-NN model
cp = classperf(y_test);
cp = classperf(cp, y_dt_pred);
modelAccuracy = cp.CorrectRate; % Model accuracy
fprintf('Decision tree model accuracy = %0.3f\n', modelAccuracy);
modelSensitivity = cp.Sensitivity; % Model sensitivity
fprintf('Decision tree model sensitivity = %0.3f\n', modelSensitivity);
modelSpecificity = cp.Specificity; % Model specificity
fprintf('Decision tree model specificity = %0.3f\n', modelSpecificity);

% Estimating area under curve
[X, Y, ~, AUC] = perfcurve(y_test, y_dt_prob(:,1), 0); % This command generates the outputs to plot the ROC curve
fprintf('Model AUC = %0.3f\n', AUC);

% Plotting the ROC curve
figure; plot(X, Y,'r-','LineWidth',2);
title('ROC curve for decision tree-based classification','FontSize',14,'FontWeight','bold');
xlabel('False positive rate','FontSize',14,'FontWeight','bold');
ylabel('True positive rate','FontSize',14,'FontWeight','bold');
set(gca,'FontWeight','bold','FontSize',14,'LineWidth',2);

%% Ensemble learning using Bagging
Mdl = fitctree(x_train, y_train);
model = fitcensemble(x_train, y_train, 'Method', 'Bag');
[y_dt_pred,y_dt_prob] = predict(model, x_test);

% Post processing 
cp = classperf(y_test);
cp = classperf(cp, y_dt_pred);


modelAccuracy = cp.CorrectRate; % Model accuracy 
fprintf('Bagging model accuracy = %0.3f\n', modelAccuracy); 
modelSensitivity = cp.Sensitivity; % Model sensitivity 
fprintf('Bagging model sensitivity = %0.3f\n', modelSensitivity);
modelSpecificity = cp.Specificity; % Model specificity 
fprintf('Bagging model specificity = %0.3f\n', modelSpecificity);

% Estimating area under curve
[X, Y, ~, AUC] = perfcurve(y_test, y_dt_prob(:,1), 0); % This command generates the outputs to plot the ROC curve 
fprintf('Model AUC = %0.3f\n', AUC); 

% Plotting the ROC curve 
figure; plot(X, Y,'r-','LineWidth',2); 
title('ROC curve for decision tree-based classification','FontSize',14,'FontWeight','bold');
xlabel('False positive rate','FontSize',14,'FontWeight','bold'); 
ylabel('True positive rate','FontSize',14,'FontWeight','bold'); 
set(gca,'FontWeight','bold','FontSize',14,'LineWidth',2);

%% Part II: only Data from images
% Read data
fprintf('Reading data for model training...\n');

info = importdata('ADNI_features.csv');
names = info.textdata;
info_data = info.data;
%read image names for merging
img_folder = 'images/';
img_path_list = dir(strcat(img_folder, '*.png'));
img_num = length(img_path_list);

labels = zeros(img_num, 1);
%read from images
for i = 1:img_num
    img_name = img_path_list(i).name;
    img = imread(strcat(img_folder, img_name));
    img = double(img);
    images(i, :) = img(:);
    img_name_no_suffix = img_name(1:end-4);
    idx = find(strcmp(names, img_name_no_suffix));
    labels(i, 1) = info_data(idx-1, 1);
    new_data(i,:) = raw_data(idx-1,:); %for part 3
end

%devide test and train data
ratio = 0.8;
num_train = round(img_num*ratio);

img_train = images(1:num_train, :);
img_test = images(num_train+1:end, :);

labels_train = labels(1:num_train, 1);
labels_test = labels(num_train+1:end, :);

img_train = img_train';
img_test = img_test';
%% PCA

fprintf('Running PCA...\n');
[evectors_train, score_train, evalues_train] = pca(img_train');
[evectors_test, score_test, evalues_test] = pca(img_test');

num_eigenvalues = 80;
evectors_train = evectors_train(:, 1:num_eigenvalues);
evectors_test = evectors_test(:, 1:num_eigenvalues);

mean_train = mean(img_train, 2);
shifted_images_train = img_train - repmat(mean_train, 1, num_train);
mean_test = mean(img_test, 2);
shifted_images_test = img_test - repmat(mean_test, 1, img_num-num_train);

features_train = evectors_train' * shifted_images_train;
features_test = evectors_test' * shifted_images_test;

% normalize the data
features_train = normalize(features_train);
features_test = normalize(features_test);
%% Model training and testing
% I have used SVM for classification

fprintf('Training and testing the model...\n');
model = fitcsvm(features_train', labels_train);
[svm_pred, svm_prob] = predict(model, features_test');

%% Post processing SVM model

fprintf('Evaluating model performance...\n');
cp = classperf(labels_test);
cp = classperf(cp, svm_pred);

modelAccuracy = cp.CorrectRate; % Model accuracy 
fprintf('Model accuracy = %0.3f\n', modelAccuracy); 
modelSensitivity = cp.Sensitivity; % Model sensitivity 
fprintf('Model sensitivity = %0.3f\n', modelSensitivity);
modelSpecificity = cp.Specificity; % Model specificity 
fprintf('Model specificity = %0.3f\n', modelSpecificity);

%% Plotting the ROC curve 

[X, Y, ~, AUC] = perfcurve(labels_test, svm_prob(:,1), 1); % This command generates the outputs to plot the ROC curve 
fprintf('Model AUC = %0.3f\n', AUC); 
plot(X, Y,'b-','LineWidth',1); hold on;
title('ROC curve for SVM','FontSize',14,'FontWeight','bold');
xlabel('False positive rate','FontSize',14,'FontWeight','bold'); 
ylabel('True positive rate','FontSize',14,'FontWeight','bold'); 
set(gca,'FontWeight','bold','FontSize',14,'LineWidth',2);

%% feature selection
%Univariate feature ranking for regression using F-tests
[idx,scores] = fsrftest(features_train', labels_train);
find(isinf(scores))
bar(scores(idx))
xlabel('Predictor rank')
ylabel('Predictor importance score')
j = idx(1:50);
X = features_train';
Y = labels_train;
Z=features_test';
% get the most 50 important features
X1 = [];
Z1 = [];
for i = 1:50
    s=j(1,i);
    X1 = [X1 X(:,s)]; 
    Z1 = [Z1 Z(:,s)];
end
%% Ensemble learning using Bagging
model = fitcensemble(X1,Y, 'Method', 'Bag');
[y_dt_pred,y_dt_prob] = predict(model, Z1);

% Post processing 
cp = classperf(labels_test);
cp = classperf(cp, y_dt_pred);

modelAccuracy = cp.CorrectRate; % Model accuracy 
fprintf('Bagging model accuracy = %0.3f\n', modelAccuracy); 
modelSensitivity = cp.Sensitivity; % Model sensitivity 
fprintf('Bagging model sensitivity = %0.3f\n', modelSensitivity);
modelSpecificity = cp.Specificity; % Model specificity 
fprintf('Bagging model specificity = %0.3f\n', modelSpecificity);

% Estimating area under curve
[X, Y, ~, AUC] = perfcurve(labels_test, y_dt_prob(:,1), 0); % This command generates the outputs to plot the ROC curve 
fprintf('Model AUC = %0.3f\n', AUC); 

% Plotting the ROC curve 
figure; plot(X, Y,'r-','LineWidth',2); 
title('ROC curve for decision tree-based classification','FontSize',14,'FontWeight','bold');
xlabel('False positive rate','FontSize',14,'FontWeight','bold'); 
ylabel('True positive rate','FontSize',14,'FontWeight','bold'); 
set(gca,'FontWeight','bold','FontSize',14,'LineWidth',2);
%% feature selection
%Univariate feature ranking for regression using F-tests
[idx,scores] = fsrftest(features_train', labels_train);
find(isinf(scores))
bar(scores(idx))
xlabel('Predictor rank')
ylabel('Predictor importance score')
j = idx(1:50);
X = features_train';
Y = labels_train;
Z=features_test';
% get the most 50 important features
X1 = [];
Z1 = [];
for i = 1:50
    s=j(1,i);
    X1 = [X1 X(:,s)]; 
    Z1 = [Z1 Z(:,s)];
end
%% Ensemble learning using Bagging
model = fitcensemble(X1,Y, 'Method', 'Bag');
[y_dt_pred,y_dt_prob] = predict(model, Z1);

% Post processing 
cp = classperf(labels_test);
cp = classperf(cp, y_dt_pred);

modelAccuracy = cp.CorrectRate; % Model accuracy 
fprintf('Bagging model accuracy = %0.3f\n', modelAccuracy); 
modelSensitivity = cp.Sensitivity; % Model sensitivity 
fprintf('Bagging model sensitivity = %0.3f\n', modelSensitivity);
modelSpecificity = cp.Specificity; % Model specificity 
fprintf('Bagging model specificity = %0.3f\n', modelSpecificity);

% Estimating area under curve
[X, Y, ~, AUC] = perfcurve(labels_test, y_dt_prob(:,1), 0); % This command generates the outputs to plot the ROC curve 
fprintf('Model AUC = %0.3f\n', AUC); 

% Plotting the ROC curve 
figure; plot(X, Y,'r-','LineWidth',2); 
title('ROC curve for decision tree-based classification','FontSize',14,'FontWeight','bold');
xlabel('False positive rate','FontSize',14,'FontWeight','bold'); 
ylabel('True positive rate','FontSize',14,'FontWeight','bold'); 
set(gca,'FontWeight','bold','FontSize',14,'LineWidth',2);
%% part III : combination of images data and csv data

%combine matrix
new_data(:,1) = [];
T=[images new_data];

%devide test and train data
ratio = 0.8;
num_train = round(img_num*ratio);

img_train = T(1:num_train, :);
img_test = T(num_train+1:end, :);

labels_train = labels(1:num_train, 1);
labels_test = labels(num_train+1:end, :);

img_train = img_train';
img_test = img_test';
%% PCA

fprintf('Running PCA...\n');
[evectors_train, score_train, evalues_train] = pca(img_train');
[evectors_test, score_test, evalues_test] = pca(img_test');

num_eigenvalues = 80;
evectors_train = evectors_train(:, 1:num_eigenvalues);
evectors_test = evectors_test(:, 1:num_eigenvalues);

mean_train = mean(img_train, 2);
shifted_images_train = img_train - repmat(mean_train, 1, num_train);
mean_test = mean(img_test, 2);
shifted_images_test = img_test - repmat(mean_test, 1, img_num-num_train);

features_train = evectors_train' * shifted_images_train;
features_test = evectors_test' * shifted_images_test;

%% Model training and testing
% normalize the data
features_train = normalize(features_train);
features_test = normalize(features_test);
%% Plot: Visualize the data representation in the space of the first three principal components.
scatter3(features_train(:,1),features_train(:,2),features_train(:,3))
axis equal
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
zlabel('3rd Principal Component')
%% create model
fprintf('Training and testing the model...\n');
model = fitcsvm(features_train', labels_train);
[svm_pred, svm_prob] = predict(model, features_test');

%% Post processing SVM model

fprintf('Evaluating model performance...\n');
cp = classperf(labels_test);
cp = classperf(cp, svm_pred);

modelAccuracy = cp.CorrectRate; % Model accuracy 
fprintf('Model accuracy = %0.3f\n', modelAccuracy); 
modelSensitivity = cp.Sensitivity; % Model sensitivity 
fprintf('Model sensitivity = %0.3f\n', modelSensitivity);
modelSpecificity = cp.Specificity; % Model specificity 
fprintf('Model specificity = %0.3f\n', modelSpecificity);

%% Plotting the ROC curve 

[X, Y, ~, AUC] = perfcurve(labels_test, svm_prob(:,1), 1); % This command generates the outputs to plot the ROC curve 
fprintf('Model AUC = %0.3f\n', AUC); 
plot(X, Y,'b-','LineWidth',1); hold on;
title('ROC curve for SVM','FontSize',14,'FontWeight','bold');
xlabel('False positive rate','FontSize',14,'FontWeight','bold'); 
ylabel('True positive rate','FontSize',14,'FontWeight','bold'); 
set(gca,'FontWeight','bold','FontSize',14,'LineWidth',2);


%% optimize hyperparameters automatically using fitrsvm
rng default
Mdl = fitrsvm(features_train', labels_train,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'))
%The output is the regression with the minimum estimated cross-validation loss.
%% plot 
X = normalize(features_train');
y = normalize(labels_train);
[svm_pred, svm_prob]  = predict(model, features_test');
svm_pred1 = normalize(svm_pred)
labels_test1 = normalize(labels_test)
h1 = plot(X,y,'o')
hold on
h2 = plot(labels_test1,svm_pred1,'x')
legend('Data','Predictions')

%% DT
rng(1); % For reproducibility
MdlDefault = fitctree(features_train', labels_train,'CrossVal','on');
numBranches = @(x)sum(x.IsBranch);
mdlDefaultNumSplits = cellfun(numBranches, MdlDefault.Trained);

figure;
histogram(mdlDefaultNumSplits)
view(MdlDefault.Trained{1},'Mode','graph')
%% The number of splits is around 9
X = features_train';
Y = labels_train;
Mdl6 = fitctree(X,Y,'MaxNumSplits',9,'CrossVal','on');
view(Mdl6.Trained{1},'Mode','graph')
classErrorDefault = kfoldLoss(MdlDefault)
classError6 = kfoldLoss(Mdl6)
%Mdl15 is much less complex and performs better than MdlDefault.

%% DT
X = features_train';
Y = labels_train;
Z=features_test';
Mdl = fitctree(X,Y,'OptimizeHyperparameters','auto')
%% 
X1=[X(:,1), X(:,16),X(:,38),X(:,3), X(:,13),X(:,28),X(:,42), X(:,2)]
Z1=[Z(:,1), Z(:,16),Z(:,38),Z(:,3), Z(:,13),Z(:,28),Z(:,42), Z(:,2)]
%% dt
[~,y_dt_prob] = predict(Mdl, features_test');
[~,y_dt_pred] = max(y_dt_prob);
%[y_dt_pred,y_dt_prob] = predict(Mdl, x_test);

% indices = crossvalind('Kfold',labels_test,10);
% Post processing 
cp = classperf(labels_test);
%cp = classperf(cp, y_dt_pred);

modelAccuracy = cp.CorrectRate; % Model accuracy 
fprintf('Decision tree model accuracy = %0.3f\n', modelAccuracy); 
modelSensitivity = cp.Sensitivity; % Model sensitivity 
fprintf('Decision tree model sensitivity = %0.3f\n', modelSensitivity);
modelSpecificity = cp.Specificity; % Model specificity 
fprintf('Decision tree model specificity = %0.3f\n', modelSpecificity);

% Estimating area under curve
[X, Y, ~, AUC] = perfcurve(labels_test, y_dt_prob(:,1), 0); % This command generates the outputs to plot the ROC curve 
fprintf('Model AUC = %0.3f\n', AUC); 

% Plotting the ROC curve 
figure; plot(X, Y,'r-','LineWidth',2); 
title('ROC curve for decision tree-based classification','FontSize',14,'FontWeight','bold');
xlabel('False positive rate','FontSize',14,'FontWeight','bold'); 
ylabel('True positive rate','FontSize',14,'FontWeight','bold'); 
set(gca,'FontWeight','bold','FontSize',14,'LineWidth',2);
%% feature selection
%Univariate feature ranking for regression using F-tests
[idx,scores] = fsrftest(features_train', labels_train);
find(isinf(scores))
bar(scores(idx))
xlabel('Predictor rank')
ylabel('Predictor importance score')
j = idx(1:50);
X = features_train';
Y = labels_train;
Z=features_test';
% get the most 50 important features
X1 = [];
Z1 = [];
for i = 1:50
    s=j(1,i);
    X1 = [X1 X(:,s)]; 
    Z1 = [Z1 Z(:,s)];
end
%% Ensemble learning using Bagging
model = fitcensemble(X1,Y, 'Method', 'Bag');
[y_dt_pred,y_dt_prob] = predict(model, Z1);

% Post processing 
cp = classperf(labels_test);
cp = classperf(cp, y_dt_pred);

modelAccuracy = cp.CorrectRate; % Model accuracy 
fprintf('Bagging model accuracy = %0.3f\n', modelAccuracy); 
modelSensitivity = cp.Sensitivity; % Model sensitivity 
fprintf('Bagging model sensitivity = %0.3f\n', modelSensitivity);
modelSpecificity = cp.Specificity; % Model specificity 
fprintf('Bagging model specificity = %0.3f\n', modelSpecificity);

% Estimating area under curve
[X, Y, ~, AUC] = perfcurve(labels_test, y_dt_prob(:,1), 0); % This command generates the outputs to plot the ROC curve 
fprintf('Model AUC = %0.3f\n', AUC); 

% Plotting the ROC curve 
figure; plot(X, Y,'r-','LineWidth',2); 
title('ROC curve for decision tree-based classification','FontSize',14,'FontWeight','bold');
xlabel('False positive rate','FontSize',14,'FontWeight','bold'); 
ylabel('True positive rate','FontSize',14,'FontWeight','bold'); 
set(gca,'FontWeight','bold','FontSize',14,'LineWidth',2);
%% feature selection
%Univariate feature ranking for regression using F-tests
[idx,scores] = fsrftest(features_train', labels_train);
find(isinf(scores))
bar(scores(idx))
xlabel('Predictor rank')
ylabel('Predictor importance score')
j = idx(1:50);
X = features_train';
Y = labels_train;
Z=features_test';
% get the most 50 important features
X1 = [];
Z1 = [];
for i = 1:50
    s=j(1,i);
    X1 = [X1 X(:,s)]; 
    Z1 = [Z1 Z(:,s)];
end
%% Decision tree

model = fitctree(X1, Y);
[~,y_dt_prob] = predict(model, Z1);
[~,y_dt_pred] = max(y_dt_prob');

% Post processing 
cp = classperf(labels_test);
cp = classperf(cp, y_dt_pred);
modelAccuracy = cp.CorrectRate; % Model accuracy 
fprintf('Decision tree model accuracy = %0.3f\n', modelAccuracy); 
modelSensitivity = cp.Sensitivity; % Model sensitivity 
fprintf('Decision tree model sensitivity = %0.3f\n', modelSensitivity);
modelSpecificity = cp.Specificity; % Model specificity 
fprintf('Decision tree model specificity = %0.3f\n', modelSpecificity);

% Estimating area under curve
[X, Y, ~, AUC] = perfcurve(labels_test, y_dt_prob(:,1), 0); % This command generates the outputs to plot the ROC curve 
fprintf('Model AUC = %0.3f\n', AUC); 

% Plotting the ROC curve 
figure; plot(X, Y,'r-','LineWidth',2); 
title('ROC curve for decision tree-based classification','FontSize',14,'FontWeight','bold');
xlabel('False positive rate','FontSize',14,'FontWeight','bold'); 
ylabel('True positive rate','FontSize',14,'FontWeight','bold'); 
set(gca,'FontWeight','bold','FontSize',14,'LineWidth',2);

%% feature selection
%Univariate feature ranking for regression using F-tests
[idx,scores] = fsrftest(features_train', labels_train);
find(isinf(scores))
bar(scores(idx))
xlabel('Predictor rank')
ylabel('Predictor importance score')
j = idx(1:30);
X = features_train';
Y = labels_train;
Z=features_test';
% get the most 50 important features
X1 = [];
Z1 = [];
for i = 1:30
    s=j(1,i);
    X1 = [X1 X(:,s)]; 
    Z1 = [Z1 Z(:,s)];
end
%% create model
fprintf('Training and testing the model...\n');
model = fitcsvm(X1, Y);
[svm_pred, svm_prob] = predict(model, Z1);

% Post processing SVM model

fprintf('Evaluating model performance...\n');
cp = classperf(labels_test);
cp = classperf(cp, svm_pred);

modelAccuracy = cp.CorrectRate; % Model accuracy 
fprintf('Model accuracy = %0.3f\n', modelAccuracy); 
modelSensitivity = cp.Sensitivity; % Model sensitivity 
fprintf('Model sensitivity = %0.3f\n', modelSensitivity);
modelSpecificity = cp.Specificity; % Model specificity 
fprintf('Model specificity = %0.3f\n', modelSpecificity);

% Plotting the ROC curve 

[X, Y, ~, AUC] = perfcurve(labels_test, svm_prob(:,1), 0); % This command generates the outputs to plot the ROC curve 
fprintf('Model AUC = %0.3f\n', AUC); 
plot(X, Y,'b-','LineWidth',1); hold on;
title('ROC curve for SVM','FontSize',14,'FontWeight','bold');
xlabel('False positive rate','FontSize',14,'FontWeight','bold'); 
ylabel('True positive rate','FontSize',14,'FontWeight','bold'); 
set(gca,'FontWeight','bold','FontSize',14,'LineWidth',2);



