function [WAR,UAR,separability] = linearClassAnalysis(X_train,X_test,labels_train,labels_test)
% function [WAR,UAR,separability] = linearClassAnalysis(X_train,X_test,labels_train,labels_test)
%
% Calculates linear classification performance
%
% Inputs:
%   X_train         : training samples
%   X_test          : testing samples
%   labels_train    : training labels
%   labels_test     : test labels
%
% Outputs:
%   WAR             : weighted average recall (0-1)
%   UAR             : unweighted average recall (0-1)
%   separability    : average discrimination loss across all pairings
  


d = zeros(max(labels_train)).*NaN;
f = zeros(max(labels_train),1);
d_acc = zeros(max(labels_train),1).*NaN;


for class1 = 1:max(labels_train)    
    Label = zeros(sum(labels_test == class1),1);
    f(class1) = length(Label);
    % Compare class1 against all other classes
    for class2 = 1:max(labels_train)
        if(class1 ~= class2)
            x1 = X_train(labels_train == class1,:);
            x2 = X_train(labels_train == class2,:);            
            
            x1_test = X_test(labels_test == class1,:);
            x2_test = X_test(labels_test == class2,:);
            
            % Train binary SVM using the two classes  
            [Mdl,fitinfo] = fitclinear([x1;x2],[zeros(size(x1,1),1);ones(size(x2,1),1)]);
            
            % Store test loss
            d(class1,class2) = loss(Mdl,[x1_test;x2_test],[zeros(size(x1_test,1),1);ones(size(x2_test,1),1)],'LossFun','classiferr');
            
            %  Predict test samples. Output is always 0 for class1 samples that are correctly classified
            Label = Label+predict(Mdl,[x1_test]);
                        
        end
    end    
    % Measure class1 accuracy as proportion of samples that got 0 from all
    % classifications.
    d_acc(class1) = sum(Label == 0)/length(Label);    
end
fprintf('\n\n');

separability = 1-nanmean(d(:));
UAR = nanmean(d_acc(f > 0));
WAR = nansum(d_acc.*f)./nansum(f);
