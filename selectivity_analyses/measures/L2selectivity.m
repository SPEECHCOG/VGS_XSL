function [L2_mean,L2_std] = L2selectivity(Y)
% function [L2_mean,L2_std] = L2selectivity(Y)
%
% Calculates multidimensional d-prime measure for network nodes
%
%  Y = [sample x node x class]

% Find nodes units that are still alive (must have positive total
% abs(activation))
tmp = squeeze(nansum(nansum(abs(Y),1),3));
alive = tmp > 0;

d = zeros(size(Y,3)).*NaN;
for class1 = 1:size(Y,3)
    for class2 = 1:size(Y,3)
        if(class1 ~= class2)
            x1 = nanmean(Y(:,alive,class1),1);  % mean vectors
            x2 = nanmean(Y(:,alive,class2),1);
             
            d1 = nanvar(Y(:,alive,class1),[],1); % variance vectors
            d2 = nanvar(Y(:,alive,class2),[],1);
            
            normer = sqrt(0.5.*(d1+d2)); % common SD
            
            % Take RMS of neuron-specific d-primes
            d(class1,class2) = sqrt(nanmean(((x1-x2)./normer).^2));           
        end
    end
end

L2_mean = nanmean(d(:));
L2_std = nanstd(d(:));