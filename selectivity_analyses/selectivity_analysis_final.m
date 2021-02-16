
training_corpora = {'places','coco'};
target_corpora = {'brent','coco'};
modelnames = {'CNN17','CNN1','RNNres'};
% Parameters

if(isfolder('/Users/rasaneno/')) % running on local
    % where to store results
    outputdir = '/Users/rasaneno/rundata/timeanalysis_Okko/analysis_outputs/';    
    % Where Khazar's codes are
    sourcedir = '/Users/rasaneno/rundata/timeanalysis_Okko/';
    % Where model activations are
    actdir = '/Volumes/BackupHD/Khazar_temporal_segmentation/activations/';
    doplots = 1;
elseif(isfolder('/scratch/specog/')) % running on cluster
    outputdir = '/scratch/specog/timeanalysis_Okko_acts/analysis_outputs/';
    sourcedir = '/scratch/specog/timeanalysis_Okko/';
    actdir = '/scratch/specog/timeanalysis_Okko_acts/activations/';
    addpath('/home/rasaneno/ACT/');
    doplots = 0;
end


N_utt_limiter = Inf; % utterances to analyze. Set to "Inf" for all.
Ns = 50; % Sampling subset size (number of tokens per type)
representation_style = 'average'; % 'all', 'midpoint', or 'average'
normalize_activations = 0; % do mean and variance norming to activations?


for corpiter = 1:2
    training_corpus = training_corpora{corpiter};    
    for targiter = 1:2
        target_corpus = target_corpora{targiter};
        
        corpus = sprintf('%s-%s',training_corpus,target_corpus);
        
        fprintf('\n#######\nProcessing %s.\n#######\n',corpus);
             
        N_models = length(modelnames);
        
        if(strcmp(training_corpus,'places'))
            target_len = 1024;
        else
            target_len = 512;
        end
        
        for model = 1:3 % choose model here among the options listed above
            
            modelname = modelnames{model};
            
            fprintf('\nModel: %s.\n\n',modelname);
            
            % Note: I have named layers as layer_0.mat, layer_1.mat etc., so they
            % always come out in correct order with dir().
            
            % Data locations
            datadir = sprintf('%s/%s/%s/layers/',actdir,corpus,modelname);
            datadir_random = sprintf('%s/%s/%s_random/layers/',actdir,corpus,modelname);
                        
            % Parse annotations
                                   
            if(contains(target_corpus,'brent'))
                % Load zero-padding information
                padfile = [sourcedir '/output/step_2/brent/zero_pad_len.mat'];
                onsets = load(padfile);
                onsets = double(onsets.all_zeropad_lens);
                
                if(strcmp(training_corpus,'coco')) % fix zero padding lengths for coco-based models
                    onsets = onsets-512;
                end
                
                if(isfolder('/Users/rasaneno/'))
                    annofile = '/Users/rasaneno/rundata/timeanalysis_Okko/brent_anno.mat';
                elseif(isfolder('/scratch/specog/'))
                    annofile = '/scratch/specog/timeanalysis_Okko_acts/brent_anno.mat';
                end
                                
                load(annofile);
                N_utt = min(N_utt_limiter,length(anno.filename));
                ref.files = anno.filename(1:N_utt);
                ref.syllables.name = anno.syllables(1:N_utt);
                ref.syllables.onset = anno.t_onset_syllable(1:N_utt);
                ref.syllables.offset = anno.t_offset_syllable(1:N_utt);
                ref.words.name = anno.words(1:N_utt);
                ref.words.onset = anno.t_onset_word(1:N_utt);
                ref.words.offset = anno.t_offset_word(1:N_utt);
                ref.phones.name = anno.phones(1:N_utt);
                ref.phones.onset = anno.t_onset_phones(1:N_utt);
                ref.phones.offset = anno.t_offset_phones(1:N_utt);
                
                % Fix timestamps to be in frames
                for k = 1:length(ref.phones.name)
                    ref.phones.name{k} = cellstr(ref.phones.name{k});
                    ref.phones.onset{k} = round(ref.phones.onset{k}.*100);
                    ref.phones.offset{k} = round(ref.phones.offset{k}.*100);
                    ref.syllables.onset{k} = round(ref.syllables.onset{k}.*100);
                    ref.syllables.offset{k} = round(ref.syllables.offset{k}.*100);
                    ref.words.name{k} = cellstr(ref.words.name{k});
                    ref.words.onset{k} = round(ref.words.onset{k}.*100);
                    ref.words.offset{k} = round(ref.words.offset{k}.*100);
                end
                                
            elseif(contains(target_corpus,'coco'))
                
                annofile = [sourcedir '/output/step_6/coco/validation_onsets_3.mat'];
                load(annofile);
                                
                padfile = [sourcedir '/output/step_2/coco/zero_pad_test_3.mat'];
                load(padfile);
                onsets = double(all_zeropad_lens);
                
                if(strcmp(training_corpus,'coco'))  % fix zero padding lengths for coco-based models
                    onsets = onsets-512;
                end
                N_utt = min(N_utt_limiter,length(wavfile_names));
                ref.files = wavfile_names(1:N_utt);
                ref.syllables.name = syllables(1:N_utt);
                ref.syllables.onset = t_onset_syllables(1:N_utt);
                ref.syllables.offset = t_offset_syllables(1:N_utt);
                ref.words.name = words(1:N_utt);
                ref.words.onset = t_onset_words(1:N_utt);
                ref.words.offset = t_offset_words(1:N_utt);
                ref.phones.name = phones(1:N_utt);
                ref.phones.onset = t_onset_phones(1:N_utt);
                ref.phones.offset = t_offset_phones(1:N_utt);
                
                % Fix timestamps to be in frames
                for k = 1:length(ref.phones.name)
                    ref.phones.name{k} = cellstr(ref.phones.name{k});
                    ref.phones.onset{k} = round(ref.phones.onset{k}./10);
                    ref.phones.offset{k} = round(ref.phones.offset{k}./10);
                    ref.syllables.name{k} = cellstr(ref.syllables.name{k});
                    ref.syllables.onset{k} = round(ref.syllables.onset{k}./10);
                    ref.syllables.offset{k} = round(ref.syllables.offset{k}./10);
                    ref.words.name{k} = cellstr(ref.words.name{k});
                    ref.words.onset{k} = round(ref.words.onset{k}./10);
                    ref.words.offset{k} = round(ref.words.offset{k}./10);
                end
            end
            
            a = dir([datadir '/*.mat']);
            
            
            N_layers = length(a);
            
            WAR_KNN = zeros(N_layers,3,2);
            UAR_KNN = zeros(N_layers,3,2);
            WAR_lin = zeros(N_layers,3,2);
            UAR_lin = zeros(N_layers,3,2);
            psi_mean = zeros(N_layers,3,2);
            psi_std = zeros(N_layers,3,2);
            psi_max = zeros(N_layers,3,2);
            psi_max_std = zeros(N_layers,3,2);
            L2_mean = zeros(N_layers,3,2);
            L2_std = zeros(N_layers,3,2);
            purity = zeros(N_layers,3,2);
            purity_random = zeros(N_layers,3,2);
            purity_random_sd = zeros(N_layers,3,2);                        
            separability = zeros(N_layers,3,2);
            cross_dist = zeros(N_layers,3,2);
            within_dist = zeros(N_layers,3,2);
            separability_lin = zeros(N_layers,3,2);            
            unique_units = cell(3,1);
            unit_counts = cell(3,1);
            
            
            
            YY = {};
            
            units = {'phones','syllables','words'};
            
            for random = 0:1 % iterate over trained and non-trained models
                
                fprintf('\nRandom = %d.\n',random);
                if(random == 0)
                    a = dir([datadir '/*.mat']);
                else
                    a = dir([datadir_random '/*.mat']);
                end
                
                for layer = 1:N_layers % iterate through layers
                    fprintf('\nLayer: %d.\n',layer-1);
                    % Load activations
                    D = load([a(layer).folder '/' a(layer).name]);
                    sample = D.filters;
                    
                    for u_iter = 1:3 % Iterate across phones, syllables and words
                        
                        unit = units{u_iter};
                        
                        if(strcmp(unit,'phones'))
                            unit_names = ref.phones.name;
                            unit_onsets = ref.phones.onset;
                            unit_offsets = ref.phones.offset;
                        elseif(strcmp(unit,'syllables'))
                            unit_names = ref.syllables.name;
                            unit_onsets = ref.syllables.onset;
                            unit_offsets = ref.syllables.offset;
                            
                        elseif(strcmp(unit,'words'))
                            unit_names = ref.words.name;
                            unit_onsets = ref.words.onset;
                            unit_offsets = ref.words.offset;
                        end
                        
                        % Get unique phones and corresponding frequencies
                        uq_units = {};
                        uq_counts = zeros(10000,1);
                        for k = 1:N_utt
                            uq_units = cellunion(uq_units,unique(unit_names{k}));
                            for j = 1:length(unit_names{k})
                                uq_counts(strcmp(uq_units,unit_names{k}{j})) = uq_counts(strcmp(uq_units,unit_names{k}{j}))+1;
                            end
                        end
                        
                        % Sort by count and take all those that occur more than "Ns"
                        % times.
                        uq_counts = uq_counts(1:length(uq_units));
                        [uq_counts,i] = sort(uq_counts,'descend');
                        uq_units = uq_units(i);
                        
                        i = find(uq_counts < Ns,1);
                        uq_units = uq_units(1:i-1);
                        uq_counts = uq_counts(1:i-1);
                        
                        unique_units{u_iter} = uq_units;
                        unit_counts{u_iter} = uq_counts;
                        
                        
                        
                        % Extract labeled activations
                        featdim = size(sample,3);
                        X = zeros(target_len*N_utt,featdim);
                        labels = zeros(target_len*N_utt,1);
                        utt_id = zeros(target_len*N_utt,1);
                        
                        rng(123,'twister'); % Reset random seed here to ensure same tokens for each analysis
                        
                        % Shuffle utterance order to avoid any local corpus structure
                        utt_order = randperm(N_utt);
                        
                        wloc = 1;
                        for k = utt_order
                            % some reshaping and type conversion
                            y = squeeze(sample(k,:,:));
                            y = double(y);
                            
                            % Upsample to target length if needed (deep
                            % layers)
                            if(size(y,1) < target_len)
                                upsample_ratio = round(target_len/size(y,1));
                                y = repelem(y,upsample_ratio,1);
                                if(size(y,1) > target_len)
                                    y = y(1:target_len,:);
                                end
                            end
                            
                            % get frame-level labels
                            local_labels = zeros(target_len,1);
                            for j = 1:length(unit_offsets{k})
                                phone_id = find(strcmp(uq_units,unit_names{k}{j})); % get phone ID number
                                if(~isempty(phone_id))                                    
                                    phone_start =round(unit_onsets{k}(j))+onsets(k); % start time in frames
                                    phone_end = round(unit_offsets{k}(j))+onsets(k); % end time in frames
                                    if(phone_end <= target_len && phone_start > 0)
                                        local_labels(phone_start:phone_end) = phone_id;
                                    end
                                end
                            end
                            
                            % Add current utterance to full data pool
                            
                            if(strcmp(representation_style,'all'))
                                X(wloc:wloc+target_len-1,:) = y;
                                labels(wloc:wloc+target_len-1) = local_labels;
                                utt_id(wloc:wloc+target_len-1) = k;
                                wloc = wloc+target_len;
                            elseif(strcmp(representation_style,'midpoint'))
                                cc = findchangepoints(local_labels)+1; % onsets of each phone
                                midpoints = round(cc(1:end-1)+diff(cc)/2);
                                
                                for j = 1:length(midpoints)
                                    if(local_labels(midpoints(j)) ~= 0)
                                        X(wloc,:) = y(midpoints(j),:);
                                        labels(wloc) = local_labels(midpoints(j));
                                        utt_id(wloc) = k;
                                        wloc = wloc+1;
                                    end
                                end
                                
                            elseif(strcmp(representation_style,'average'))
                                cc = findchangepoints(local_labels)+1; % onsets of each phone
                                midpoints = round(cc(1:end-1)+diff(cc)/2);
                                for j = 1:length(cc)-1
                                    X(wloc,:) = mean(y(cc(j):cc(j+1),:));
                                    labels(wloc) = local_labels(midpoints(j));
                                    utt_id(wloc) = k;
                                    wloc = wloc+1;
                                end
                            end
                        end
                        
                        % Remove unused memery slots
                        X = X(1:wloc-1,:);
                        labels = labels(1:wloc-1);
                        utt_id = utt_id(1:wloc-1);
                        
                        labels = double(labels);
                        
                        % Remove any unlabeled frames (zero pads + trailing/leading silences)
                        torem = labels == 0;
                        labels(torem) = [];
                        X(torem,:) = [];
                        utt_id(torem) = [];
                        
                        % Split frames randomly to train and test
                        
                        totrain = 0.8;     % proportion of data put to train
                        N = size(X,1);
                        
                        % Create indices for training and testing utterances
                        train_inds = round(1:N*totrain);
                        % make sure split to train/test happens at utterance level
                        tmp = find(diff(utt_id(train_inds(end):end)) ~= 0,1);
                        train_inds = round(1:N*totrain+tmp-1);
                        test_inds = round(N*totrain+tmp:N);
                        
                        % sanity check: training and testing utterances must be
                        % different
                        if(~isempty(intersect(utt_id(train_inds),utt_id(test_inds))))
                            disp('something went wrong with dividing data into train and test');
                        end
                        
                        % Extract training and testing samples and their labels for
                        % classification experiments
                        X_train = X(train_inds,:);
                        labels_train = labels(train_inds);
                        X_test = X(test_inds,:);
                        labels_test = labels(test_inds);
                        
                        % Normalize activations with training data statistics?
                        if(normalize_activations)
                            mm1 = mean(X_train);
                            dd1 = std(X_train);
                            
                            X_train = X_train-repmat(mm1,size(X_train,1),1);
                            X_train = X_train./repmat(dd1,size(X_train,1),1);
                            
                            X_test = X_test-repmat(mm1,size(X_test,1),1);
                            X_test = X_test./repmat(dd1,size(X_test,1),1);
                        end
                        
                        % Fix potential NaNs from activations
                        X_train(isnan(X_train)) = 0;
                        X_test(isnan(X_test)) = 0;
                        
                        
                        % Subsample Ns tokens for each type
                        X_sample = X;
                        labels_sample = labels;
                        
                        ii = randperm(length(labels)); % some further shuffling
                        X_sample = X_sample(ii,:);
                        labels_sample = labels_sample(ii);
                        
                        count = zeros(length(uq_units),1);
                        Y = zeros(Ns,size(X,2),length(uq_units)).*NaN; % sample x dim x class
                        for k = 1:length(uq_units)
                            tmp = find(labels_sample == k,Ns);
                            Y(1:length(tmp),:,k) = X_sample(tmp,:);
                            count(k) = length(tmp);
                        end
                        
                        % This shouldn't do anything anymore as thresholding is already done
                        % earlier
                        tmp = find(count < Ns);
                        Y(:,:,tmp) = [];
                        
                        % Normalize sub-sample?
                        if(normalize_activations)
                            meme = nanmean(nanmean(Y,1),3);
                            dev = nanmean(nanstd(Y,[],1),3);
                            Y = Y-repmat(meme,[size(Y,1) 1 size(Y,3)]);
                            Y = Y./repmat(dev,[size(Y,1) 1 size(Y,3)]);
                        end
                        
                        
                        YY{layer,u_iter,random+1} = Y; % Store the sub-sample
                        
                        % PSI
                        fprintf('Running PSI\n');
                        [psi_mean(layer,u_iter,random+1),psi_std(layer,u_iter,random+1),psi_max(layer,u_iter,random+1),psi_max_std(layer,u_iter,random+1)] = PSI(Y);
                        
                        % Separability
                        fprintf('Running separability analysis\n');
                        [separability(layer,u_iter,random+1),cross_dist(layer,u_iter,random+1),within_dist(layer,u_iter,random+1)] = separabilityAnalysis(Y);
                        
                        % Multidim D-prime analysis
                        fprintf('Running d-prime analysis\n');
                        [L2_mean(layer,u_iter,random+1),L2_std(layer,u_iter,random+1)] = L2selectivity(Y);
                        
                        % KNN classification (non-linear)
                        fprintf('Running KNN classification\n');
                        [WAR_KNN(layer,u_iter,random+1),UAR_KNN(layer,u_iter,random+1)] = KNNAnalysis(X_train,X_test,labels_train,labels_test,15);
                        
                        % SVM classification (linear)
                        fprintf('Running SVM classification\n');
                        [WAR_lin(layer,u_iter,random+1),UAR_lin(layer,u_iter,random+1),separability_lin(layer,u_iter,random+1)] = linearClassAnalysis(X_train,X_test,labels_train,labels_test);
                        
                        % K-means clustering
                        fprintf('Running kmeans classification\n');
                        [purity(layer,u_iter,random+1),purity_random(layer,u_iter,random+1),purity_random_sd(layer,u_iter,random+1)] = kmeans_analysis(Y);
                        
                        fprintf('UAR_KNN %f in layer %d\n',UAR_KNN(layer,u_iter,random+1),layer-1);
                    end
                end
            end
            
            results.unique_units = unique_units;
            results.unit_counts = unit_counts;
            results.YY = YY;
            results.psi_mean = psi_mean;
            results.psi_std = psi_std;
            results.psi_max = psi_max;
            results.psi_max_std = psi_max_std;
            results.separability = separability;
            results.cross_dist = cross_dist;
            results.within_dist = within_dist;
            results.L2_mean = L2_mean;
            results.L2_std = L2_std;
            results.WAR_KNN = WAR_KNN;
            results.UAR_KNN = UAR_KNN;
            results.WAR_lin = WAR_lin;
            results.UAR_lin = UAR_lin;
            results.purity = purity;
            results.purity_random = purity_random;
            results.purity_random_sd = purity_random_sd;
            
            
            filename = sprintf('%s/%s_%s_%s_%d_%s_norm%d',outputdir,training_corpus,target_corpus,modelname,N_utt_limiter,representation_style,normalize_activations);
            
            save(filename,'results');
            
            % Some plotting
            colors = [0    0.4470    0.7410;
                0.8500    0.3250    0.0980;
                0.9290    0.6940    0.1250];
            
            if(doplots)
                h = figure(6);clf;
                for unit = 1:3
                    subplot(4,2,1);hold on;plot(0:N_layers-1,WAR_KNN(:,unit,1)*100,'LineWidth',2,'Color',colors(unit,:));xlabel('layer');ylabel('WAR (%)');grid;title('KNN WAR');set(gca,'XTickLabel',0:N_layers-1);set(gca,'XTick',0:N_layers-1);
                    subplot(4,2,2);hold on;plot(0:N_layers-1,UAR_KNN(:,unit,1)*100,'LineWidth',2,'Color',colors(unit,:));xlabel('layer');ylabel('UAR (%)');grid;title('KNN UAR');set(gca,'XTickLabel',0:N_layers-1);set(gca,'XTick',0:N_layers-1);
                    subplot(4,2,3);hold on;plot(0:N_layers-1,WAR_lin(:,unit,1)*100,'LineWidth',2,'Color',colors(unit,:));xlabel('layer');ylabel('WAR (%)');grid;title('linear WAR');set(gca,'XTickLabel',0:N_layers-1);set(gca,'XTick',0:N_layers-1);
                    subplot(4,2,4);hold on;plot(0:N_layers-1,UAR_lin(:,unit,1)*100,'LineWidth',2,'Color',colors(unit,:));xlabel('layer');ylabel('UAR (%)');grid;title('linear UAR');set(gca,'XTickLabel',0:N_layers-1);set(gca,'XTick',0:N_layers-1);
                    subplot(4,2,5);hold on;plot(0:N_layers-1,psi_mean(:,unit,1),'LineWidth',2,'Color',colors(unit,:));xlabel('layer');ylabel('PSI');grid;title('PSI mean');set(gca,'XTickLabel',0:N_layers-1);set(gca,'XTick',0:N_layers-1);
                    subplot(4,2,6);hold on;plot(0:N_layers-1,psi_max(:,unit,1),'LineWidth',2,'Color',colors(unit,:));xlabel('layer');ylabel('PSI');grid;title('PSI max');set(gca,'XTickLabel',0:N_layers-1);set(gca,'XTick',0:N_layers-1);
                    subplot(4,2,7);hold on;plot(0:N_layers-1,L2_mean(:,unit,1),'LineWidth',2,'Color',colors(unit,:));xlabel('layer');ylabel('d-prime');grid;title('multidim. d-prime');set(gca,'XTickLabel',0:N_layers-1);set(gca,'XTick',0:N_layers-1);
                    subplot(4,2,8);hold on;plot(0:N_layers-1,-separability(:,unit,1),'LineWidth',2,'Color',colors(unit,:));xlabel('layer');ylabel('d_d_i_f_f-d_s_a_m_e');grid;title('cosine separability');set(gca,'XTickLabel',0:N_layers-1);set(gca,'XTick',0:N_layers-1);
                end
                
                for unit = 1:3
                    subplot(4,2,1);hold on;plot(0:N_layers-1,WAR_KNN(:,unit,2)*100,'LineWidth',2,'LineStyle','--','Color',colors(unit,:));xlabel('layer');ylabel('WAR (%)');grid;title('KNN WAR');set(gca,'XTickLabel',0:N_layers-1);grid;
                    subplot(4,2,2);hold on;plot(0:N_layers-1,UAR_KNN(:,unit,2)*100,'LineWidth',2,'LineStyle','--','Color',colors(unit,:));xlabel('layer');ylabel('UAR (%)');grid;title('KNN UAR');set(gca,'XTickLabel',0:N_layers-1);grid;
                    subplot(4,2,3);hold on;plot(0:N_layers-1,WAR_lin(:,unit,2)*100,'LineWidth',2,'LineStyle','--','Color',colors(unit,:));xlabel('layer');ylabel('WAR (%)');grid;title('linear WAR');set(gca,'XTickLabel',0:N_layers-1);grid;
                    subplot(4,2,4);hold on;plot(0:N_layers-1,UAR_lin(:,unit,2)*100,'LineWidth',2,'LineStyle','--','Color',colors(unit,:));xlabel('layer');ylabel('UAR (%)');grid;title('linear UAR');set(gca,'XTickLabel',0:N_layers-1);grid;
                    subplot(4,2,5);hold on;plot(0:N_layers-1,psi_mean(:,unit,2),'LineWidth',2,'LineStyle','--','Color',colors(unit,:));xlabel('layer');ylabel('PSI');grid;title('PSI mean');set(gca,'XTickLabel',0:N_layers-1);grid;
                    subplot(4,2,6);hold on;plot(0:N_layers-1,psi_max(:,unit,2),'LineWidth',2,'LineStyle','--','Color',colors(unit,:));xlabel('layer');ylabel('PSI');grid;title('PSI max');set(gca,'XTickLabel',0:N_layers-1);grid;
                    subplot(4,2,7);hold on;plot(0:N_layers-1,L2_mean(:,unit,2),'LineWidth',2,'LineStyle','--','Color',colors(unit,:));xlabel('layer');ylabel('d-prime');grid;title('multidim. d-prime');set(gca,'XTickLabel',0:N_layers-1);grid;
                    subplot(4,2,8);hold on;plot(0:N_layers-1,-separability(:,unit,2),'LineWidth',2,'LineStyle','--','Color',colors(unit,:));xlabel('layer');ylabel('d_d_i_f_f-d_s_a_m_e');grid;title('cosine separability');set(gca,'XTickLabel',0:N_layers-1);grid;
                end
                legend('Location','Northeast',units);
                drawnow;
            end
        end
    end
end
