function [purity,purity_random,purity_random_sd] = kmeans_analysis(Y)
% function [purity,purity_random,purity_random_sd] = kmeans_analysis(Y)
%
% Performs k-means based clustering analysis for data in Y.
% 
%  Y = [sample x node x class]

doPCA = 0;

YZ = zeros(numel(Y)./size(Y,2),size(Y,2));

labs = zeros(size(YZ,1),1);
wloc = 1;
for k = 1:size(Y,3)
    YZ(wloc:wloc+size(Y,1)-1,:) = Y(:,:,k);
    labs(wloc:wloc+size(Y,1)-1) = k;    
    wloc = wloc+size(Y,1);
end

Y(:,sum(YZ) == 0,:) = [];

YZ(:,sum(YZ) == 0) = [];

if(doPCA)
    [coeff,score,latent,tsquare,explained] = pca(YZ);
    
    explained = cumsum(explained./sum(explained));
    tmp = find(explained > 0.99,1);
    
    if(isempty(tmp))
        tmp = length(explained);
    end
    
    coeff = coeff(:,1:tmp);
    
    YZ_pca = YZ*coeff;
    
    Y_pca = zeros(size(Y,1),size(YZ_pca,2),size(Y,3));
    
    for k = 1:size(Y,3)
        Y_pca(:,:,k) = Y(:,:,k)*coeff;
    end
end

Y(isnan(Y)) = 0;
Y(isinf(Y)) = 0;
YZ(isnan(YZ)) = 0;
YZ(isinf(YZ)) = 0;

Y_pca = Y;
YZ_pca = YZ;

% K-means using class means as centroids

means = squeeze(mean(Y_pca,1));
[idx,c] = kmeans(YZ_pca,size(Y,3),'Start',means','Distance','sqeuclidean');


class_size = size(Y,1);
class_purity = zeros(size(Y,3),1);
k = 1;
for wloc = 1:class_size:length(idx)-class_size+1    
    hypos = idx(wloc:wloc+class_size-1);    
    class_purity(k) = sum(hypos == k)./class_size;    
    k = k+1;
end

purity = nanmean(class_purity);


% K-means using randomly initialized centroids
purity_iter = zeros(5,1);

for iter = 1:5
    
    [idx,c] = kmeans(YZ_pca,size(Y,3),'Distance','sqeuclidean');
    class_size = size(Y,1);
    
    % Confusion matrix
    ref = repelem(1:size(Y,3),1,class_size)';        
    CC = zeros(max(ref));
    for k = 1:length(idx)
        CC(ref(k),idx(k)) = CC(ref(k),idx(k))+1;
    end
    
    % Measure overall purity by assigning each cluster to represent one
    % type.
    class_purity = zeros(size(Y,3),1);
    for c = 1:size(Y,3)
        
        [row,col] = find(CC == max(CC(:)));
        true = row(1);
        hypo = col(1);
        class_purity(true) = CC(true,hypo)./class_size;
        CC(true,:) = 0;
        CC(:,hypo) = 0;
    end
    
    purity_iter(iter) = mean(class_purity);
    
end

purity_random = mean(purity_iter);
purity_random_sd = std(purity_iter);




