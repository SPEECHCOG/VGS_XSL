function vec_out = linear_boundary_regression(sample,onsets,ref,target_len)
% function vec_out = linear_boundary_regression(sample,onsets,ref,target_len)
%
% Create ANN activation representations with linear regression from
% activations to target signals encoding linguistic unit boundaries

D = sample; % activations from sample
onsets = max(1,onsets); % zero padding info

vec_out = zeros(size(sample,1),target_len,3);

for unit = 1:3 % iterate through phones, syllables and words
    
    X_all = zeros(1000000,size(sample,3));
    Y_all = zeros(1000000,1);

    N = size(D,1);
    
    loc = 1;
    for signal = 1:2500
        
        
        y = squeeze(D(signal,:,:)); % get activations for this utterance
        
        % Select correct annotated unit boundaries for the given unit type
        if(unit == 1)
            try
                bb = [ref.phones.onset{signal} ref.phones.offset{signal}(end)];
            catch
                bb = [ref.phones.onset{signal};ref.phones.offset{signal}(end)];
            end
        elseif(unit == 2)
            try
                bb = [ref.syllables.onset{signal} ref.syllables.offset{signal}(end)];
            catch
                bb = [ref.syllables.onset{signal};ref.syllables.offset{signal}(end)];
            end
        elseif(unit == 3)
            try
                bb = [ref.words.onset{signal} ref.words.offset{signal}(end)];
            catch
                bb = [ref.words.onset{signal};ref.words.offset{signal}(end)];
            end
        end
        bb = unique(bb);
        bb(bb <= 0) = 1;
        
        onset_current = onsets(signal);
               
        
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
        
        y = y(onset_current+1:end,:);

        X_all(loc:loc+size(y,1)-1,:) = y;
        
        % Create target signal envelope with Gaussian kernels centered at 
        % each boundary 
        
        targ = zeros(size(y,1),1);
        
        % Target size depends on the unit (+-20 ms or +-40 ms)
        if(unit == 1)
            sigma = 2/1.96;
        elseif(unit > 1)
            sigma = 2;
        end
        
        for k = 1:length(bb)            
            tmp = normpdf(1:length(targ),bb(k),sigma)';            
            tmp = tmp./max(tmp);            
            targ = targ+tmp;
            
        end
        
        targ(targ > 1) = 1; % Clip max amplitude to 1 if overlapping Gaussians
                
        Y_all(loc:loc+length(targ)-1) = targ;
        
        loc = loc+length(targ);
        
    end
    
    % Now all activations and targets are stored in X_all and Y_all
        
    Y_all = Y_all(1:loc-1,:);
    X_all = X_all(1:loc-1,:);
    
    Y_all(isnan(Y_all)) = 0;
    X_all(isnan(X_all)) = 0;
    
    
    % Train a predictor model using pseudoinverse    
    W = pinv(X_all)*Y_all;
    
    
    % Create predicted target envelopes using the regression model
    for signal = 1:N
        
        y = squeeze(sample(signal,:,:));
        y = double(y);
        
        % Resample length if needed
        if(size(y,1) < target_len)
            rat = round(target_len/size(y,1));
            y = repelem(y,rat,1);
            if(size(y,1) > target_len)
                y = y(1:target_len,:);
            end
        else
            rat = 0;
        end
        
        pred_env = y*W;
        
        pred_env(1:onsets(signal)) = 0;
        
        vec_out(signal,:,unit) = pred_env;
        
    end
    
end








