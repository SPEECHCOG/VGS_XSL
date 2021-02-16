function [separability,cross_dist,within_dist] = separabilityAnalysis(Y)
% function [separability,cross_dist,within_dist] = separabilityAnalysis(Y)
%
% Caulculates average distance between tokens from the same type, and
% between tokens from different types. Returns the average within- minus cross-type
% distance and the individual averages.
% 
% Y is of shape [sample x node x class]

% Find units that are still alive
tmp = squeeze(nansum(nansum(abs(Y),1),3));
alive = tmp > 0;

metric = 'Cosine';

% Convert from tensor to matrix format
Y_full = zeros(size(Y,1)*size(Y,3),size(Y,2));
y_label = zeros(size(Y,1)*size(Y,3),1);

wloc = 1;
for k = 1:size(Y,3)
   Y_full(wloc:wloc+size(Y,1)-1,:) = Y(:,:,k);
   y_label(wloc:wloc+size(Y,1)-1) = k;
   wloc = wloc+size(Y,1);
end

% calculate cross-class and within class distances
cross_dist = zeros(size(Y,3),1);
within_dist = zeros(size(Y,3),1);
i_full = 1:size(Y_full,1);
for class = 1:size(Y,3)    
    i_class = find(y_label == class);
    i_nonclass = setxor(i_class,i_full);
    
    D_cross = pdist2(Y_full(i_class,alive),Y_full(i_nonclass,alive),metric);
    cross_dist(class) = nanmean(D_cross(:));
    
    D_within = pdist2(Y_full(i_class,alive),Y_full(i_class,alive),metric);
    for jj = 1:size(D_within,1)
        D_within(jj,jj) = NaN;
    end
    within_dist(class) = nanmean(D_within(:));    
end

separability = nanmean(within_dist-cross_dist);
cross_dist = nanmean(cross_dist);
within_dist = nanmean(within_dist);

