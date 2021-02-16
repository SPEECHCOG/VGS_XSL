
training_corpora = {'places','coco'};
target_corpora = {'brent','coco'};
modelnames = {'CNN17','CNN1','RNNres'};
newmodelnames = {'CNN0','CNN1','RNN'}; % redefine model names for plotting
units = {'phones','syllables','words'};
N_models = length(modelnames);


% where are the analysis results?
resultdir = '/Users/rasaneno/rundata/timeanalysis_Okko/analysis_outputs/'; 

N_utt_limiter = Inf; % utterances to analyze. Set to "Inf" for all.
Ns = 50; % Sampling subset size (number of tokens per type)
representation_style = 'average'; % 'all', 'midpoint', or 'average'
normalize_activations = 0; % do mean and variance norming to activations?

set(0,'defaultaxesfontsize',18);
set(0,'defaulttextfontsize',18);

for corpiter = 1:2
    training_corpus = training_corpora{corpiter};
    for targiter = 1:2
        target_corpus = target_corpora{targiter};
        
        corpus = sprintf('%s-%s',training_corpus,target_corpus);
        
        fprintf('\n#######\nProcessing %s.\n#######\n',corpus);
        h = figure('Position',[337 82 1500 950]);clf;
        x = 0;
        for model = 1:3 % choose model here among the options listed above
            modelname = modelnames{model};
            newmodelname = newmodelnames{model};
            
            filename = sprintf('%s/%s_%s_%s_%d_%s_norm%d_kmeans.mat',resultdir,training_corpus,target_corpus,modelname,N_utt_limiter,representation_style,normalize_activations);
            
            if(exist(filename,'file'))
                load(filename)
            else
                error('Given data file does not exist');
            end
            
            % Some plotting
            colors = [0    0.4470    0.7410;
                0.8500    0.3250    0.0980;
                0.9290    0.6940    0.1250];
            
            N_layers = size(results.psi_mean,1);
            
            
            for unit = 1:3
                
                subplot(3,4,4+x);hold on;plot(0:N_layers-1,results.UAR_KNN(:,unit,1)*100,'LineWidth',2,'Marker','*','Color',colors(unit,:));xlabel('layer');ylabel('UAR (%)');grid;set(gca,'XTickLabel',0:N_layers-1);set(gca,'XTick',0:N_layers-1);
                if(x < 4)
                    title(sprintf('non-linear separability'));
                end
                if(targiter == 1)
                    ylim([0 50])
                else
                    ylim([0 90])
                end
                
                subplot(3,4,3+x);hold on;plot(0:N_layers-1,results.UAR_lin(:,unit,1)*100,'LineWidth',2,'Marker','*','Color',colors(unit,:));xlabel('layer');ylabel('UAR (%)');grid;set(gca,'XTickLabel',0:N_layers-1);set(gca,'XTick',0:N_layers-1);
                if(x < 4)
                    title(sprintf('linear separability'));
                end
                
                if(targiter == 1)
                    ylim([0 50])
                else
                    ylim([0 90])
                end
                
                subplot(3,4,1+x);hold on;plot(0:N_layers-1,results.L2_mean(:,unit,1),'LineWidth',2,'Marker','*','Color',colors(unit,:));xlabel('layer');ylabel('d-prime');grid;set(gca,'XTickLabel',0:N_layers-1);set(gca,'XTick',0:N_layers-1);ylim([0.15 1.2]);
                if(x < 4)
                    title(sprintf('node separability'));
                end
                
                
                subplot(3,4,2+x);drawstds(h,0:N_layers-1,results.purity_random(:,unit,1),results.purity_random_sd(:,unit,1),0.15,1.5,[0.5 0.5 0.5]);
                subplot(3,4,2+x);hold on;plot(0:N_layers-1,results.purity_random(:,unit,1),'LineWidth',2,'Marker','*','Color',colors(unit,:));xlabel('layer');ylabel('purity');grid;set(gca,'XTickLabel',0:N_layers-1);set(gca,'XTick',0:N_layers-1);
                
                if(targiter == 1)
                    ylim([0 0.25]);
                else
                    ylim([0 0.5]);
                end
                if(x < 4)
                    title(sprintf('layer clusteredness'));
                end
                
            end
            
            for unit = 1:3
                
                subplot(3,4,4+x);hold on;plot(0:N_layers-1,results.UAR_KNN(:,unit,2)*100,'LineWidth',1.5,'LineStyle','--','Color',colors(unit,:));           
                subplot(3,4,3+x);hold on;plot(0:N_layers-1,results.UAR_lin(:,unit,2)*100,'LineWidth',1.5,'LineStyle','--','Color',colors(unit,:));                
                subplot(3,4,1+x);hold on;plot(0:N_layers-1,results.L2_mean(:,unit,2),'LineWidth',1.5,'LineStyle','--','Color',colors(unit,:));
                
                   if(unit == 1)
                    tt = text(0,1,newmodelname,'Units','normalized');
                    tt.Rotation = 90;
                    tt.FontSize = 20;
                    tt.Position = [-0.3 0.5 0];
                    tt.HorizontalAlignment = 'center';
                    tt.FontWeight = 'bold';
                end
                                
                subplot(3,4,2+x);drawstds(h,0:N_layers-1,results.purity_random(:,unit,2),results.purity_random_sd(:,unit,2),0.15,1.5,[0.5 0.5 0.5]);
                subplot(3,4,2+x);hold on;plot(0:N_layers-1,results.purity_random(:,unit,2),'LineWidth',1.5,'LineStyle','--','Color',colors(unit,:));
                               
            end
            
            
            ll = {};
            ll{1} = 'L0';
            for j = 2:N_layers
                ll{j} = sprintf('L%d',j-1);
            end
            
            
            for ss = 1:4
                subplot(3,4,ss+x);
                set(gca,'XTickLabel',ll);
                xlim([-0.5 N_layers-0.5])
            end
            
            if(model == 3)
            subplot(3,4,11);
            tt = text(0,1,corpus,'Units','normalized');
            tt.FontSize = 20;
            tt.Position = [-0.1979 -0.3535 0];
            tt.HorizontalAlignment = 'center';
            tt.FontWeight = 'bold';
            end
            
            if(model == 3)
                subplot(3,4,4+x);
                tmp=legend('Location','Northwest',units);
                tmp.Position = [0.4852 0.9232 0.0693 0.0684];
            end
            
            
            drawnow;
            x = x+4;
        end
        drawnow;
        [~,ff] = fileparts(filename);
        teekuva([ff 'final']);
    end
end

% Create table of unit statistics as well

TABLE = {};

TABLE{1,2} = target_corpora{1};
TABLE{1,4} = target_corpora{2};
TABLE{2,2} = 'N types';
TABLE{2,3} = 'N tokens';
TABLE{2,4} = 'N types';
TABLE{2,5} = 'N tokens';
TABLE{3,1} = 'phones';
TABLE{4,1} = 'syllables';
TABLE{5,1} = 'words';

for corpiter = 1
    training_corpus = training_corpora{corpiter};
    for targiter = 1:2
        target_corpus = target_corpora{targiter};
        
        corpus = sprintf('%s-%s',training_corpus,target_corpus);        
        fprintf('\n#######\nProcessing %s.\n#######\n',corpus);                
        for model = 1 % choose model here among the options listed above
            modelname = modelnames{model};
            
            filename = sprintf('%s/%s_%s_%s_%d_%s_norm%d.mat',resultdir,training_corpus,target_corpus,modelname,N_utt_limiter,representation_style,normalize_activations);
            
            if(exist(filename,'file'))
                load(filename)
            else
                error('Given data file does not exist');
            end
            
            for unit = 1:3
                
                N_types = length(results.unique_units{unit});
                N_tokens = sum(results.unit_counts{unit});
                
                TABLE{unit+2,2+(targiter-1)*2} = N_types;
                TABLE{unit+2,3+(targiter-1)*2} = N_tokens;
                
            end
            
        end
    end
end




