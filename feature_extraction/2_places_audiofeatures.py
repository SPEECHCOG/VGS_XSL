import os
import scipy.io
import scipy,scipy.io
import numpy 
import librosa

path_project = '.../VGS_XSL/'
path_speech = '.../PlacesAudio_400k_distro/'

path_in = os.path.join(path_project ,'output/step_1/places/')
file_in = 'meta_shuffling.mat'

path_out =  os.path.join(path_project ,'output/step_2/places/')


###############################################################################


outfilenames = ['logmel_val','logmel_train1','logmel_train2','logmel_train3','logmel_train4','logmel_train5','logmel_train6','logmel_train7','logmel_train8','logmel_test']

metadata_splitting = scipy.io.loadmat(path_in + 'meta_shuffling.mat' , variable_names = ['inds_all','inds_all_shuffled','inds_test','inds_val','inds_train'])
inds_train = metadata_splitting ['inds_train'][0]
inds_val = metadata_splitting ['inds_val'][0]
inds_test = metadata_splitting ['inds_test'][0]

n_val = len(inds_val)
n_test = len(inds_test)
n_train = len(inds_train)

inds_chunks = [inds_val]
for i in range(8):
    inds_chunks.append(inds_train[i*5000:(i+1)*50000])
inds_chunks.append(inds_test)
#.............................................................Reading data file

data = scipy.io.loadmat(path_in + file_in, variable_names = ['all_captions', 'all_imnames', 'all_speakerids','all_wavnames']) 
all_wavnames = data['all_wavnames']

temp = all_wavnames
all_wavnames = []

for element_wav in temp:  
    correct_element = element_wav.strip()        
    all_wavnames.append(correct_element)  
del temp

###############################################################################
                                 # FEATURE EXTRACTION #
###############################################################################
#.............................................................................. Audio features parameters
nb_mel_bands = 40  

win_len_time = 0.025
win_hop_time = 0.01
sr_target = 16000

win_len_sample = int (sr_target * win_len_time)
win_hop_sample = int (sr_target * win_hop_time)
nfft = win_len_sample


logmel_all = []

############################################################################### Extracting audio features

for chunk in range(10):
    print('..... chunk number is ........' + str(chunk+1))
    outfilename = outfilenames[chunk]
    inds_chunk = inds_chunks[chunk]
    list_of_wav_files = [all_wavnames[item] for item in inds_chunk]
    count = 0

    for wav_file in list_of_wav_files:
       
        audio_file_name = path_speech  + wav_file 
        
        y, sr = librosa.load(audio_file_name)
        y = librosa.core.resample(y, sr, sr_target) 
        #..........................................................................     
        mel_feature = librosa.feature.melspectrogram(y=y, sr=sr_target, n_fft=win_len_sample, hop_length=win_hop_sample, n_mels=nb_mel_bands,power=2.0)
        
        #........................................  removing zeros from mel features
        zeros_mel = mel_feature[mel_feature==0]          
        if numpy.size(zeros_mel)!= 0:
            #print('### there are zeros in mel feature ............' + str(wav_file))
            mel_flat = mel_feature.flatten('F')
            mel_temp =[value for counter, value in enumerate(mel_flat) if value!=0]
        
            if numpy.size(mel_temp)!=0:
                min_mel = numpy.min(numpy.abs(mel_temp))
            else:
                min_mel = 1e-12 
               
            mel_feature[mel_feature==0] = min_mel
        #.......................................................................... Preparing final vectors
        
     
        logmel_feature = numpy.transpose(10*numpy.log10(mel_feature))   
            
        logmel_all.append(logmel_feature)
        
        print(count)
        count+= 1
        
    import pickle
    filename = path_out + outfilename
    outfile = open(filename,'wb')
    pickle.dump(logmel_all ,outfile)  



