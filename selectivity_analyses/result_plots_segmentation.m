
training_corpora = {'places','coco'};
target_corpora = {'brent','coco'};
modelnames = {'CNN17','CNN1','RNNres'};
newmodelnames = {'CNN0','CNN1','RNN'};
units = {'phones','syllables','words'};

% Parameters
N_models = length(modelnames);

% Where are the analysis results?
resultdir = '/Users/rasaneno/rundata/timeanalysis_Okko/analysis_outputs/segmentation/';

N_utt_limiter = Inf; % utterances to analyze. Set to "Inf" for all.
Ns = 50; % Sampling subset size (number of tokens per type)
representation_style = 'average'; % 'all', 'midpoint', or 'average'
normalize_activations = 0; % do mean and variance norming to activations?
usepeaks = 1;
usediff = 0;
doexample = 0;
donorm = 1;
DoG_filter = 1;


for corpiter = 1:2
    training_corpus = training_corpora{corpiter};
    for targiter = 1:2
        target_corpus = target_corpora{targiter};
        
        corpus = sprintf('%s-%s',training_corpus,target_corpus);
        
        methods = {'entropy','L2'};
        ff = {'phones','syllables','words'};
        
        for method_iter = 1:2
            
            
            barcols = zeros(7,3);
            barcols(1,:) = [1 1 1];
            for k = 2:7
                barcols(k,:) = barcols(k-1,:).*8/9;
            end
            
            
            modelname = modelnames{end};
            method = methods{method_iter};
            
            filename = sprintf('%s/%s_%s_%s_%d_%s_norm%d_segmentation_davidfilter_%d_usediff_%d_usepeaks_%d_final.mat',resultdir,training_corpus,target_corpus,modelname,N_utt_limiter,method,donorm,DoG_filter,usediff,usepeaks);
            
            load(filename);
            
            
            % Plot selectivity as a function of layers
            rah = figure(123);clf;hold on;
            rah.Position = [1 270 1200 500];
            
            set(0,'defaulttextfontsize',16);
            set(0,'defaultaxesfontsize',16);
            
            for model = 1:N_models
                layers_of_interest = 1:size(results.F_all{model,1},2);
                if(model == 3)
                    layers_of_interest = 1:4;
                end
                
                
                if(length(layers_of_interest) == 7)
                    d = 0.35;
                elseif(length(layers_of_interest) == 6)
                    d = 0.315;
                else
                    d = 0.275;
                end
                offset = [-d:2*d/(length(layers_of_interest)-1):d];
                
                subplot(3,3,1+3*(model-1));hold on;
                %% F-score
                % Actual model
                vv = [];
                for unittype = 1:3
                    vv = [vv;max(results.F_all{model,1}(unittype,layers_of_interest,:),[],3)];
                end
                
                doh = bar(vv);
                for k = 1:length(doh)
                    doh(k).FaceColor = barcols(k,:);
                    doh(k).EdgeColor = [0 0 0];
                end
                
                % Untrained baseline
                vv = [];
                for unittype = 1:3
                    [~,ii] = max(results.F_all{model,2}(unittype,layers_of_interest,:),[],3);
                    vv = [vv;diag(squeeze(results.F_all{model,2}(unittype,layers_of_interest,ii)))'];
                end
                hold on;
                tmp = bar(vv);
                for k = 1:length(tmp)
                    tmp(k).BarWidth = 0.2;
                    tmp(k).FaceColor = [1 0 0];
                    tmp(k).EdgeColor = [1 0 0];
                    tmp(k).FaceAlpha = 0.3;
                    tmp(k).EdgeAlpha = 0.3;
                end
                set(gca,'XTick',1:3);
                set(gca,'XTickLabel',{'phones','syllables','words'});
                ylabel('F-score');
                grid;
                
                for n = 1:3
                    for j = 1:length(layers_of_interest)
                        text(n+offset(j),-0.033*max(ylim),sprintf('L%d',j-1),'FontSize',8,'HorizontalAlignment','center')
                    end
                end
                
                
                subplot(3,3,2+3*(model-1));hold on;
                %% Precision
                % Actual model
                vv = [];
                for unittype = 1:3
                    [~,ii] = max(results.F_all{model,1}(unittype,layers_of_interest,:),[],3);
                    vv = [vv;diag(squeeze(results.PRC_all{model,1}(unittype,layers_of_interest,ii)))'];
                end
                
                doh = bar(vv);
                for k = 1:length(doh)
                    doh(k).FaceColor = barcols(k,:);
                    doh(k).EdgeColor = [0 0 0];
                end
                % Untrained baseline
                vv = [];
                for unittype = 1:3
                    [~,ii] = max(results.F_all{model,2}(unittype,layers_of_interest,:),[],3);
                    vv = [vv;diag(squeeze(results.PRC_all{model,2}(unittype,layers_of_interest,ii)))'];
                end
                title(newmodelnames{model});
                hold on;
                tmp = bar(vv);
                for k = 1:length(tmp)
                    tmp(k).BarWidth = 0.2;
                    tmp(k).FaceColor = [1 0 0];
                    tmp(k).EdgeColor = [1 0 0];
                    tmp(k).FaceAlpha = 0.3;
                    tmp(k).EdgeAlpha = 0.3;
                end
                set(gca,'XTick',1:3);
                set(gca,'XTickLabel',{'phones','syllables','words'});
                ylabel('precision');
                grid;
                
                for n = 1:3
                    for j = 1:length(layers_of_interest)
                        text(n+offset(j),-0.033*max(ylim),sprintf('L%d',j-1),'FontSize',8,'HorizontalAlignment','center')
                    end
                end
                
                subplot(3,3,3+3*(model-1));hold on;
                %% Recall
                % Actual model
                vv = [];
                for unittype = 1:3
                    [~,ii] = max(results.F_all{model,1}(unittype,layers_of_interest,:),[],3);
                    vv = [vv;diag(squeeze(results.RCL_all{model,1}(unittype,layers_of_interest,ii)))'];
                end
                
                doh = bar(vv);
                for k = 1:length(doh)
                    doh(k).FaceColor = barcols(k,:);
                    doh(k).EdgeColor = [0 0 0];
                end
                
                % Untrained baseline
                vv = [];
                for unittype = 1:3
                    [~,ii] = max(results.F_all{model,2}(unittype,layers_of_interest,:),[],3);
                    vv = [vv;diag(squeeze(results.RCL_all{model,2}(unittype,layers_of_interest,ii)))'];
                end
                hold on;
                tmp = bar(vv);
                for k = 1:length(tmp)
                    tmp(k).BarWidth = 0.2;
                    tmp(k).FaceColor = [1 0 0];
                    tmp(k).EdgeColor = [1 0 0];
                    tmp(k).FaceAlpha = 0.3;
                    tmp(k).EdgeAlpha = 0.3;
                end
                set(gca,'XTick',1:3);
                set(gca,'XTickLabel',{'phones','syllables','words'});
                ylabel('recall');
                grid;
                
                for n = 1:3
                    for j = 1:length(layers_of_interest)
                        text(n+offset(j),-0.033*max(ylim),sprintf('L%d',j-1),'FontSize',8,'HorizontalAlignment','center')
                    end
                end
            end
            
            drawnow;
            teekuva(sprintf('unit_segmentation_perf_%s_peaks_%d_diff_%d_%s_new',corpus,usepeaks,usediff,method));
        end
    end
end



