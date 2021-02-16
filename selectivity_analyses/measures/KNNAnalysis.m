function [WAR,UAR] = KNNAnalysis(X_train,X_test,labels_train,labels_test,k_nearest)    
% function [WAR,UAR] = KNNAnalysis(X_train,X_test,labels_train,labels_test,k_nearest)    
%
% Calculates non-linear classification performance with a KNN classifier
%
% Inputs:
%   X_train         : training samples
%   X_test          : testing samples
%   labels_train    : training labels
%   labels_test     : test labels
%   k_nearest       : how many KNN neighbors to consider
%
% Outputs:
%   WAR             : weighted average recall (0-1)
%   UAR             : unweighted average recall (0-1)
  
if nargin <5
    k_nearest = 15;
end

hypos = zeros(size(labels_test));

chunksize = 1000;   % how many test samples to test at a time
                    % as limited by memory requirements of the distance matrix

wloc =1;
while wloc < size(X_test,1)
    
    real_chunksize = min(chunksize,size(X_test,1)-wloc+1); % last chunk might be smaller than default size
    
    % distance between training and testing samples
    d = pdist2(X_test(wloc:wloc+real_chunksize-1,:),X_train,'Cosine');
    
    % sort by distance to each training sample
    [dist,ind] = sort(d,2,'ascend');
    
    % choose k nearest training samples for each test sample
    ind = ind(:,1:k_nearest);
    
    % for each test sample, get mode of corresponding k-nearest
    % training sample labels
    hypos(wloc:wloc+real_chunksize-1) = mode(labels_train(ind),2);
    
    wloc = wloc+chunksize;
    
end

% Create a confusion matrix
CC = zeros(max(max(labels_test),max(labels_train)));

for k = 1:length(hypos)
    CC(labels_test(k),hypos(k)) = CC(labels_test(k),hypos(k))+1;
end

% frequencies
f = sum(CC,2);

% normalize rows to get class specific confusion probabiliites
ss = sum(CC,2);
CC = CC./repmat(ss,1,size(CC,2));

WAR = sum(hypos == labels_test)./length(hypos);

% UAR is the mean of the diagonal of the normalized confusion matrix
tmp = diag(CC);
UAR = nanmean(tmp(f > 0));
