function yTest_miss = kNN_miss(x_train_miss, y_train_miss, missMaskTrain, x_test_miss, missMaskTest, alpha, k, prob_miss_train, num_clss)
%kNN classifier for the case of Missing Features

%find the distances Dist
% upper = min(min([x_train_miss;x_test_miss]));
dist = zeros(size(x_test_miss,1),size(x_train_miss,1));
pen = zeros(size(x_test_miss,1),size(x_train_miss,1));
for i = 1:size(x_test_miss,1)
    for j = 1:size(x_train_miss,1)
        dist(i,j) = sqrt(sum((x_train_miss(j,(missMaskTrain(j,:)==0)&(missMaskTest(i,:)==0)) - x_test_miss(i,(missMaskTrain(j,:)==0)&(missMaskTest(i,:)==0))).^2));
%         pen(i,j) = sum(prob_miss_train(xor(missMaskTrain(j,:),missMaskTrain(i,:))==1))/sum(prob_miss_train((missMaskTrain(j,:)&missMaskTest(i,:))==0));
        pen(i,j) = sum(prob_miss_train((missMaskTrain(j,:)|missMaskTest(i,:))==1))/sum(prob_miss_train);
    end
end
dist = (dist - min(min(dist)))./(max(max(dist)) - min(min(dist)));
Dist = (1-alpha)*dist + alpha*pen;
%sort the points according to Dist and find kNN
[~, Idx_sort] = sort(Dist,2,'ascend');
kNN_labels = zeros(size(x_test_miss,1),k);
for i = 1:size(x_test_miss,1)
    kNN_labels(i,:) = y_train_miss(Idx_sort(i,1:k));
end
%find majority label
edges = zeros(1,num_clss);
for i = 1:num_clss
   edges(i) = i;
end
[~,yTest_miss] = max(histc(kNN_labels,edges,2),[],2);

end

