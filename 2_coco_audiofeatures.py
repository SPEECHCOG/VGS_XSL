import os
import scipy.io
import scipy,scipy.io
import numpy 
import librosa


path_images = ".../MSCOCO/val2014/val2014"
path_speech = ".../SPEECH-COCO/val2014/val2014/wav"

path_project = '.../VGS_XSL/'

path_in = os.path.join(path_project ,'output/step_1/coco/')
file_in = 'processed_data_list.mat'

path_out =  os.path.join(path_project ,'output/step_2/coco/')
file_out = ['logmel_train1','logmel_train2','logmel_train3','logmel_train4','logmel_train5']


###############################################################################

audio_files_data = scipy.io.loadmat(path_in + file_in, variable_names=['list_of_wav_files','list_of_wav_counts'])
list_of_wav_files = audio_files_data['list_of_wav_files'][0]
list_of_wav_counts = audio_files_data['list_of_wav_counts'][0]


temp = list_of_wav_files
list_of_wav_files = []
for item in temp:
    temp_all_chunks = item[0:5]
    all_chunks = []
    for element in temp_all_chunks:
        correct_element = element.strip()    
        all_chunks.append(correct_element)
    list_of_wav_files.append(all_chunks)
del temp   

   
number_of_wavfiles = len(list_of_wav_files) * 5   
number_of_images = len(list_of_wav_files) 
n_chunks = 5 

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

############################################################################### Extracting audio features


############################################################################### Extracting audio features
import pickle

for ind_chunk in range(n_chunks):
    print('..... chunk number is ........' + str(ind_chunk+1))
    outfilename = file_out[ind_chunk]
    
    train_chunk = [item[ind_chunk] for item in list_of_wav_files]
    count = 0
    
    logmel_all = []
    for wav_file in train_chunk:
       
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
        #.......................................................................... Preparing final vector           
        logmel_feature = numpy.transpose(10*numpy.log10(mel_feature))   
            
        logmel_all.append(logmel_feature)
        
        print(count)
        count+= 1
        
    
    filename = path_out + outfilename
    outfile = open(filename,'wb')
    pickle.dump(logmel_all ,outfile)  
