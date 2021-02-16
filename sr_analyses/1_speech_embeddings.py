import numpy 
import pickle
import scipy,scipy.io

w2vfile = '.../GoogleNews-vectors-negative300.bin'
spellcheck_path = ''

path_audiomodel = ''

path_out = '.../output/step_5/embeddings/'
###############################################################################
                        # Loading Test Data #
###############################################################################
############################################################################### if Places 10k test set
def preparX (dict_logmel):
    number_of_audios = numpy.shape(dict_logmel)[0]
    number_of_audio_features = numpy.shape(dict_logmel[0])[1]
    len_of_longest_sequence = 1024

    X = numpy.zeros((number_of_audios ,len_of_longest_sequence, number_of_audio_features),dtype ='float32')


    for k in numpy.arange(number_of_audios):
        logmel_item = dict_logmel[k]
        logmel_item = logmel_item[0:len_of_longest_sequence]
        X[k,len_of_longest_sequence-len(logmel_item):, :] = logmel_item
    return X

datadir = ''
filename = datadir + 'logmel_test' 
infile = open(filename ,'rb')
logmel = pickle.load(infile)
infile.close()
Xdata = preparX (logmel)
del logmel

############################################################################### spell check
filename = 'places_testset_spellcheck.mat'                        
data =  scipy.io.loadmat(spellcheck_path + filename,
                                  variable_names=['ind_correct'])   

ind_correct = data['ind_correct'][0] 
Xdata = Xdata[ind_correct]

number_of_audios = Xdata.shape[0]
len_of_longest_sequence = Xdata.shape[1]
number_of_audio_features = Xdata.shape[2]

###############################################################################

                                # MODEL #

###############################################################################
import keras

audio_model = keras.models.load_model(path_audiomodel)
audio_model.load_weights(path_audiomodel + 'audiomodel_weights.h5')
audio_embeddings = audio_model.predict(Xdata) 
###############################################################################
                    # Finding closest distances  #
###############################################################################
import scipy.spatial as ss
top = 50
worst = 50
dist_sorted_top = []
dist_sorted_worst = []
dist_sorted_random = []                    
for audio_number in numpy.arange(number_of_audios):
    print(audio_number)
    q = audio_embeddings[audio_number:audio_number + 1]
    r = audio_embeddings
    dist_q = ss.distance.cdist(q,r, 'cosine')
    dist_q = dist_q[0]
    dist_sorted_ind = numpy.argsort(dist_q) 
    del dist_q 
    del q
    del r
    
    dist_sorted_top_q = dist_sorted_ind [0:top + 1]
    dist_sorted_worst_q = dist_sorted_ind [-(worst):]
    dist_sorted_random_q = numpy.random.choice(dist_sorted_ind,worst)

    dist_sorted_top.append(dist_sorted_top_q)
    dist_sorted_worst.append(dist_sorted_worst_q)
    dist_sorted_random.append(dist_sorted_random_q)
    
    del dist_sorted_ind 
    del dist_sorted_top_q 
    del dist_sorted_worst_q 
    del dist_sorted_random_q 
    
    #print('.............................' + str(audio_number))

###############################################################################
                    # Saving the results #
###############################################################################
                    
filename = 'sorted_dist_top' 
numpy.save(path_out + filename , dist_sorted_top )

filename = 'sorted_dist_worst' 
numpy.save(path_out + filename , dist_sorted_worst )

filename = 'sorted_dist_rand' 
numpy.save(path_out + filename , dist_sorted_random )









