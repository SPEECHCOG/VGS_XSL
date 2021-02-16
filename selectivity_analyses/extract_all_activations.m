
% Extract model activations for each training/testing combination and for
% all three models, for all layers in each model, and before and after
% training.
for trainingcorpus = {'places','coco'}
    for testingcorpus = {'brent','coco'}
        for model = {'CNN17','CNN1','RNNres'}
            for trainflag = {'trained','0'}
                ss = sprintf('/anaconda3/bin/python /Users/rasaneno/Documents/koodit/dev/khazar_temporal_analysis/new_Jan2021/model_activation_extraction/activations_unified.py %s %s %s %s',model{1},trainingcorpus{1},testingcorpus{1},trainflag{1});                               
                system(ss);                
            end           
        end        
    end
end

