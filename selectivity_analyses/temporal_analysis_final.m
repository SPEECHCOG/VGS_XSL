% Runs segmentation analyses for model activation data


training_corpora = {'places','coco'};
target_corpora = {'brent','coco'};
modelnames = {'CNN17','CNN1','RNNres'};

N_utt_limiter = Inf; % utterances to analyze. Set to "Inf" for all.

usepeaks = 1;   % Detect peaks (1) or valleys (0)?
donorm = 1;     % utterance-level z-score norm temporal representations?
DoG_filter = 1; % use difference of gaussian from Harwath et al. (2019)? Otherwise regular 1st order timediff
usediff = 0;    % set to 0 if using DoG_filter 

if(isfolder('/Users/rasaneno/')) % calculation on local
    % Where to save results
    outputdir = '/Users/rasaneno/rundata/timeanalysis_Okko/analysis_outputs/';
    % Where are Khazar's codes (needed for zero-padding info)
    sourcedir = '/Users/rasaneno/rundata/timeanalysis_Okko/';
    % Where are the model activations
    actdir = '/Volumes/BackupHD/Khazar_temporal_segmentation/activations/';
    doplots = 1;
elseif(isfolder('/scratch/specog/')) % calculation on cluster
    outputdir = '/scratch/specog/timeanalysis_Okko_acts/analysis_outputs/segmentation/';
    sourcedir = '/scratch/specog/timeanalysis_Okko/';
    actdir = '/scratch/specog/timeanalysis_Okko_acts/activations/';
    addpath('/home/rasaneno/ACT/');
    doplots = 0;
end



for corpiter = 1:2 % iterate across training corpora
    training_corpus = training_corpora{corpiter};
    for targiter = 1:2 % iterate across testing corpora
        target_corpus = target_corpora{targiter};
        
        corpus = sprintf('%s-%s',training_corpus,target_corpus);
        
        fprintf('\n#######\nProcessing %s.\n#######\n',corpus);
        
        N_models = length(modelnames);
        
        if(strcmp(training_corpus,'places'))
            target_len = 1024;
        else
            target_len = 512;
        end
        
        methods = {'entropy','L2','linseg'};
        
        for method_iter = 1:3
            
            method = methods{method_iter};
            
            PRC_all = cell(N_models,2);
            RCL_all = cell(N_models,2);
            F_all = cell(N_models,2);
            R_all = cell(N_models,2);
            
            PRC_all_r = cell(N_models,2);
            RCL_all_r = cell(N_models,2);
            F_all_r = cell(N_models,2);
            R_all_r = cell(N_models,2);
            
            for model = 1:N_models
                
                modelname = modelnames{model};
                
                fprintf('\nModel: %s with %s.\n\n',modelname,method);
                
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
                
                % DoG filter from Harwath et al. (2019)
                filt = diff(normpdf(-3:3 ,0,0.5));
                for random = 0:1 % Iterate over real and non-trained models
                    
                    if(random == 0)
                        a = dir([datadir '/*.mat']);
                    else
                        a = dir([datadir_random '/*.mat']);
                    end
                    vec = cell(length(a),1);
                    
                    for layer = 1:length(a)
                        D = load([a(layer).folder '/' a(layer).name]);
                        
                        sample = D.filters(1:N_utt,:,:);
                        
                        N = size(sample,1);
                        
                        vec{layer} = zeros(N,target_len); % Go through cases
                        
                        
                        if(strcmp(method,'linseg'))
                            vec{layer} = linear_boundary_regression(sample,onsets,ref,target_len);
                        else
                            
                            for k = 1:N
                                % Get activation for signal k
                                y = squeeze(sample(k,:,:));
                                y = double(y);
                                
                                % Resample to target_len time frames if needed
                                if(size(y,1) < target_len)
                                    rat = round(target_len/size(y,1));
                                    y = repelem(y,rat,1);
                                    if(size(y,1) > target_len)
                                        y = y(1:target_len,:);
                                    end
                                else
                                    rat = 0;
                                end
                                
                                y_org = y;
                                
                                % Sum code
                                if(strcmp(method,'L2'))
                                    y = sqrt(sum(y.^2,2));
                                    
                                    % Harwath's version with DoG filtering
                                    if(DoG_filter)
                                        y_f = filter(filt,1,y);
                                        y_f = circshift(y_f,-length(filt)/2);
                                        y = y_f;
                                    elseif(usediff) % regular timediff
                                        y = [diff(y);0];
                                    end
                                    % Fix scale of samples preceding utterance onset
                                    y(1:max(1,onsets(k))) = y(max(1,onsets(k))+1);
                                    y(end-2:end) = y(end-3);
                                    
                                    
                                elseif(strcmp(method,'entropy'))
                                    y = y-min(y(:));
                                    y = y./repmat(nansum(y,2),1,size(y,2));
                                    y = -sum(y.*log2(y+0.00000001),2)./log2(size(y,2));
                                    
                                    if(DoG_filter)
                                        y_f = filter(filt,1,y);
                                        y_f = circshift(y_f,-length(filt)/2);
                                        y = y_f;
                                    elseif(usediff)
                                        y = [diff(y);0];
                                    end
                                    % Fix scale of samples preceding utterance onset
                                    y(1:max(1,onsets(k))) = y(max(1,onsets(k))+1);
                                    y(end-2:end) = y(end-3);
                                else
                                    error('unknown measure');
                                end
                                
                                if(rat > 0)
                                    fillen = (rat+mod(rat,2));
                                    y = filter(ones(fillen,1)./fillen,1,y);
                                    y = [y(round(fillen/2)+1:end,:);zeros(round(fillen/2),size(y,2))];
                                    y(y < 0) = 0;
                                end
                                
                                
                                % Z-score norming
                                if(donorm)
                                    if(onsets(k) > 0)
                                        y = y-nanmean(y(onsets(k):end));
                                        y = y./nanstd(y(onsets(k):end));
                                    else
                                        y = y-nanmean(y(1:end));
                                        y = y./nanstd(y(1:end));
                                        
                                    end
                                end
                                
                                % Anything before utterance onset is set to
                                % zero.
                                y(1:onsets(k)) = 0;
                                
                                vec{layer}(k,:) = y;
                            end
                        end
                    end
                    
                    N_layers = length(vec);
                    
                    ff = {'phones','syllables','words'};
                    
                    % Max deviation allowed from reference boundaries
                    allowed_d = [0.02, 0.05, 0.05].*100;
                    
                    % Detection thresholds to iterate over
                    thrvals = [0.02:0.01:0.09,0.1:0.1:2.5];
                    
                    N_hits = zeros(3,N_layers,length(thrvals));
                    N_miss = zeros(3,N_layers,length(thrvals));
                    N_ref = zeros(3,N_layers,length(thrvals));
                    N_ins = zeros(3,N_layers,length(thrvals));
                    
                    N_hits_r = zeros(3,N_layers,length(thrvals));
                    N_miss_r = zeros(3,N_layers,length(thrvals));
                    N_ins_r = zeros(3,N_layers,length(thrvals));
                    
                    
                    for layer = 1:N_layers
                        fprintf('Processing layer %d...\n',layer);
                        for k = 1:N
                            
                            for unittype = 1:3
                                
                                try % this should work for Brent
                                    t_ref = round([ref.(ff{unittype}).onset{k};ref.(ff{unittype}).offset{k}(end)])+onsets(k)-1;
                                catch % this should work for COCO
                                    t_ref = round([ref.(ff{unittype}).onset{k} ref.(ff{unittype}).offset{k}(end)])+onsets(k)-1;
                                    t_ref = t_ref';
                                end
                                if(ref_center == 1)
                                    t_ref = t_ref(1:end-1) + round(diff(t_ref).*0.5);
                                end
                                
                                t_ref(t_ref < 1) = [];
                                
                                %t_ref = t_ref+1;
                                
                                t_ref = unique(t_ref);
                                
                                if(strcmp(method,'linseg'))
                                    y = vec{layer}(k,:,unittype); % get unit specific output from linseg
                                else
                                    y = vec{layer}(k,:); 
                                end
                                
                                % Iterate through detection thresholds
                                for thriter = 1:length(thrvals)
                                    thr = thrvals(thriter);
                                    
                                    % peak picking to get boundaries
                                    [maxtab,mintab] = peakdet(y,thr);
                                    
                                    if(usepeaks)
                                        if(~isempty(maxtab))
                                            hypos = maxtab(:,1);
                                        else
                                            hypos = [];
                                        end
                                    else
                                        if(~isempty(mintab))
                                            hypos = mintab(:,1);
                                        else
                                            hypos = [];
                                        end
                                    end
                                    
                                    
                                    % Exclude hypotheses that are not during
                                    % the utterance
                                    hypos(hypos < min(t_ref)) = [];
                                    hypos(hypos > max(t_ref)) = [];
                                    
                                    % Onset and offset automatically
                                    % detected for free.
                                    hypos = [t_ref(1);hypos;t_ref(end)];
                                    hypos = unique(hypos);
                                    
                                    % Generate random segmentation for
                                    % baseline
                                    
                                    hypos_random = randi(max(t_ref)-min(t_ref),length(hypos),1)+min(t_ref);
                                    
                                    % Measure number of hits based on allowed distances
                                    
                                    hit = zeros(size(t_ref));
                                    hit_r = zeros(size(t_ref));
                                    
                                    if(~isempty(hypos))
                                        d = abs(hypos'-t_ref);
                                        d_r = abs(hypos_random'-t_ref);
                                        
                                        
                                        tmp = sort(d,2,'ascend');
                                        [~,tmp2] = sort(tmp(:,1),'ascend');
                                        
                                        for r = tmp2'
                                            [peakdist,b] = min(d(r,:));
                                            if(peakdist <= allowed_d(unittype))
                                                hit(r) = 1;
                                                d(:,b) = Inf;
                                            end
                                            
                                        end
                                        
                                        tmp = sort(d_r,2,'ascend');
                                        [~,tmp2] = sort(tmp(:,1),'ascend');
                                        
                                        for r = tmp2'
                                            [a_r,b_r] = min(d_r(r,:));
                                            if(a_r <= allowed_d(unittype))
                                                hit_r(r) = 1;
                                                d_r(:,b_r) = Inf;
                                            end
                                        end
                                    end
                                    
                                    N_hits(unittype,layer,thriter) = N_hits(unittype,layer,thriter)+sum(hit);
                                    N_miss(unittype,layer,thriter) = N_miss(unittype,layer,thriter)+sum(hit == 0);
                                    N_ins(unittype,layer,thriter) = N_ins(unittype,layer,thriter)+length(hypos)-sum(hit);
                                    N_hits_r(unittype,layer,thriter) = N_hits_r(unittype,layer,thriter)+sum(hit_r);
                                    N_miss_r(unittype,layer,thriter) = N_miss_r(unittype,layer,thriter)+sum(hit_r == 0);
                                    N_ins_r(unittype,layer,thriter) = N_ins_r(unittype,layer,thriter)+length(hypos)-sum(hit_r);
                                    N_ref(unittype,layer,thriter) = N_ref(unittype,layer,thriter)+length(hit);
                                    
                                end
                            end
                            procbar(k,N);
                        end
                        fprintf('\n\n');
                    end
                    
                    
                    %% Calculate performance metrics
                    
                    PRC_all{model,random+1} = N_hits./(N_ref+N_ins);
                    RCL_all{model,random+1} = N_hits./N_ref;
                    F_all{model,random+1} = 2.*PRC_all{model,random+1}.*RCL_all{model,random+1}./(PRC_all{model,random+1}+RCL_all{model,random+1});
                    F_all{model,random+1}(isnan(F_all{model,random+1})) = 0;
                    R_all{model,random+1} = rvalue(RCL_all{model,random+1}.*100,((N_ins+N_hits)./N_ref-1).*100);
                    
                    % Same for random boundaries
                    PRC_all_r{model,random+1} = N_hits_r./(N_ref+N_ins_r);
                    RCL_all_r{model,random+1} = N_hits_r./N_ref;
                    F_all_r{model,random+1} = 2.*PRC_all_r{model,random+1}.*RCL_all_r{model,random+1}./(PRC_all_r{model,random+1}+RCL_all_r{model,random+1});
                    F_all_r{model,random+1}(isnan(F_all_r{model,random+1})) = 0;
                    R_all_r{model,random+1} = rvalue(RCL_all_r{model,random+1}.*100,((N_ins_r+N_hits_r)./N_ref-1).*100);
                    
                end
            end
            
            results.PRC_all = PRC_all;
            results.RCL_all = RCL_all;
            results.F_all = F_all;
            results.R_all = R_all;
            results.PRC_all_r = PRC_all_r;
            results.RCL_all_r = RCL_all_r;
            results.F_all_r = F_all_r;
            results.R_all_r = R_all_r;
            
            filename = sprintf('%s/%s_%s_%s_%d_%s_norm%d_segmentation_davidfilter_%d_usediff_%d_usepeaks_%d_final',outputdir,training_corpus,target_corpus,modelname,N_utt_limiter,method,donorm,DoG_filter,usediff,usepeaks);
            
            save(filename,'results');
            
        end
        
    end
end




