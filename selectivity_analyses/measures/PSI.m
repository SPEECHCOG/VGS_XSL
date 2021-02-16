function [psi_mean,psi_std,psi_max,psi_max_std] = PSI(Y)
% function [psi_mean,psi_std,psi_max,psi_max_std] = PSI(Y)
%
% Calculates Phoneme selectivity index (PSI) 
%
% Y is of shape [sample x node x class]

alpha = 0.05;

% Find units that are still alive
tmp = squeeze(sum(sum(abs(Y),1),3));
died = tmp == 0;

% Test PSI of each node

h = ones(size(Y,3),size(Y,3),size(Y,2)).*NaN;
p = ones(size(Y,3),size(Y,3),size(Y,2)).*NaN;
t = ones(size(Y,3),size(Y,3),size(Y,2)).*NaN;

for node = 1:size(Y,2)
    if(died(node) == 0)
        for class1 = 1:size(Y,3)
            for class2 = 1:size(Y,3)
                if(class1 ~= class2)
                    [h(class1,class2,node),p(class1,class2,node),~,mix] = ttest2(Y(:,node,class1),Y(:,node,class2),alpha);
                    t(class1,class2,node) = abs(mix.tstat);
                end
            end
        end
    end
end

% Take PSI of each node

psi_mean = mean(squeeze(mean(nansum(h,2),1)))./size(h,1);
psi_std = std(squeeze(mean(nansum(h,2),1)));

% Measure "max" PSI by finding the most selective node for each phone,
% excluding many-to-one mappings.

tmp = squeeze(nansum(h,2));
psi_all = zeros(size(Y,3),1);
for j = 1:size(Y,3)
    
    maxval = max(tmp(:));
    if(~isnan(maxval))
        [row,col] = find(tmp == maxval);
        phone = row(1);
        node = col(1);
        tmp(:,node) = NaN; % exclude this node
        tmp(phone,:) = NaN; % exclude this phone
        psi_all(j) = maxval;
    else
        psi_all(j) = 0;
    end
end

psi_max = mean(psi_all)./length(psi_all);
psi_max_std = std(psi_all)./length(psi_all);



