%% Implementation of kNN Classifier for Missing Features

% Required pre-loaded data in the workspace
%-------------------------------------------
% x -> dataset, rows correspond to data points
% labels -> class labels to be used for clustering validity
% choice -> type of missingness: 1: MCAR
%                                2: MAR
%                                3: MNAR-I
%                                4: MNAR-II
% frac -> fraction of missingness

%% Initialization
alpha = input('Enter the value of the parameter ALPHA for FWPD: ');
k = input('Enter the number of nearest neighbors for kNN classification: ');

%% generating the pattern of missing features as per CHOICE
[x_miss,miss_mask,prob_miss] = missGenerator(x,frac,choice);
miss_count = sum(miss_mask,2)';

%% k-fold cross-validation (multi-class)
kn = 5; % number of partitions to be made for cross-validation
num_clss = length(unique(labels));
sIze = zeros(1,num_clss);
leftover = zeros(1,num_clss);
for j = 1:num_clss
    sIze(j) = floor(size(x(labels==j,:),1)/kn);
    leftover(j) = size(x(labels==j,:),1) - (kn * floor(size(x(labels==j,:),1)/kn));
end
sIze = repmat(sIze,kn,1);
flag = 0;
% sIze_idx = 1;
for j = 1:num_clss
    if flag==0
        sIze_idx = 1;
    else
        sIze_idx = kn;
    end
    while leftover(j) > 0
        sIze(sIze_idx,j) = sIze(sIze_idx,j) + 1;
        leftover(j) = leftover(j) - 1;
        if flag==0
            sIze_idx = sIze_idx + 1;
        else
            sIze_idx = sIze_idx - 1;
        end
    end
    flag = ~flag;
end
clearvars flag leftover;

sIze_cum = cumsum(sIze);
sIze_cum = circshift(sIze_cum,1);

rand_IDX = randperm(length(x));
xx_miss = x_miss(rand_IDX,:);
yy = labels(rand_IDX);
MissCount = miss_count(rand_IDX);
MissMask = miss_mask(rand_IDX,:);

yPred = zeros(size(labels));
for i = 1:kn
    %% extracting the multi-class training and test sets for a particular partition
    idxsR_part = [];
    y_train = []; y_test = [];
    x_train_miss = []; x_test_miss = [];
    MissCountTrain = []; MissCountTest = [];
    MissMaskTrain = []; MissMaskTest = [];
    
    for j = 1:num_clss
        idxsR_temp = rand_IDX(yy==j);
        x_temp_miss = xx_miss(yy==j,:);
        mask_temp = MissMask(yy==j,:);
        count_temp = MissCount(yy==j);
        
        idxsR_temp = circshift(idxsR_temp',-1*sIze_cum(i,j)); idxsR_temp = idxsR_temp';
        x_temp_miss = circshift(x_temp_miss,-1*sIze_cum(i,j));
        mask_temp = circshift(mask_temp,-1*sIze_cum(i,j));
        count_temp = circshift(count_temp,[0 -1*sIze_cum(i,j)]);
        
        y_temp = j * ones(size(x_temp_miss,1),1);
        
        idxsR_part = [idxsR_part, idxsR_temp(1:sIze(i,j))];   
        x_test_miss = [x_test_miss; x_temp_miss(1:sIze(i,j),:)];
        MissMaskTest = [MissMaskTest; mask_temp(1:sIze(i,j),:)];
        MissCountTest = [MissCountTest, count_temp(1:sIze(i,j))];
        y_test = [y_test; y_temp(1:sIze(i,j))];
        
        x_train_miss = [x_train_miss; x_temp_miss((sIze(i,j)+1):end,:)];
        MissMaskTrain = [MissMaskTrain; mask_temp((sIze(i,j)+1):end,:)];
        MissCountTrain = [MissCountTrain, count_temp((sIze(i,j)+1):end)];
        y_train = [y_train; y_temp((sIze(i,j)+1):end)];
    end
    
    %% kNN classification
    prob_miss_train = sum(MissMaskTrain,1)/sum(sum(MissMaskTrain));
    yPred(idxsR_part) = kNN_miss(x_train_miss, y_train, MissMaskTrain, x_test_miss, MissMaskTest, alpha, k, prob_miss_train, num_clss);
    fprintf('Finished running the kNN-FWPD classifier over partition %d.\n',i);
    
end
