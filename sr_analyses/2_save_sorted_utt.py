
import numpy
import scipy,scipy.io

model_name = 'places/CNN1/'
w2vfile = '.../GoogleNews-vectors-negative300.bin'

spellcheck_path = ''
caption_path = ''

path_in  = ''
path_out = ''   

###############################################################################
                         # Calling Embeddings #
###############################################################################    

filename = 'sorted_dist_top.npy'
sorted_dist_top = numpy.load(path_in + filename  )

filename = 'sorted_dist_worst.npy'
sorted_dist_worst = numpy.load(path_in + filename  ) 

filename = 'sorted_dist_rand.npy' 
sorted_dist_rand = numpy.load(path_in + filename )

###############################################################################
                         # captions #
############################################################################### 
                        
filename = 'places_testset_spellcheck.mat'                        
data =  scipy.io.loadmat(spellcheck_path + filename,
                                  variable_names=['ind_correct'])   

ind_correct = data['ind_correct'][0] 

################################################################################
#                    # Selected wav files + captions #
################################################################################

metadata_file = caption_path + 'places_testset_captions.mat'
data = scipy.io.loadmat(metadata_file)  
captions = data['captions']

captions = captions[ind_correct]

def extract_captions (target):
    unit_list = []
    for utterance in target:
        utterance = (utterance.strip())  
        unit_list.append((utterance))    
    return  unit_list 

caption_list = extract_captions(captions)


###############################################################################
        # removing first element from top list #
###############################################################################
        
sorted_dist_top = sorted_dist_top [:, 1:]

###############################################################################
        # Finding top similar utterances based on Embedded layer #
###############################################################################
top_counts = 50
dict_top = []
for reference_ind in range(len(sorted_dist_top)):
    reference_utterance = caption_list [reference_ind]
    dict_reference = []
    dict_reference.append(reference_utterance)

    for j in range(top_counts):
        candidate_utterance = caption_list [sorted_dist_top [reference_ind,j]]
        dict_reference.append(candidate_utterance)


    dict_top.append(dict_reference) 

low_counts = 50
dict_low = []
for reference_ind in range(len(sorted_dist_worst)):
    reference_utterance = caption_list [reference_ind]
    dict_reference = []
    dict_reference.append(reference_utterance)

    for j in range(low_counts):
        candidate_utterance = caption_list [sorted_dist_worst [reference_ind,j]]
        dict_reference.append(candidate_utterance)


    dict_low.append(dict_reference) 
    
rand_counts = 50
dict_rand = []
for reference_ind in range(len(sorted_dist_rand)):
    reference_utterance = caption_list [reference_ind]
    dict_reference = []
    dict_reference.append(reference_utterance)

    for j in range(rand_counts):
        candidate_utterance = caption_list [sorted_dist_rand [reference_ind,j]]
        dict_reference.append(candidate_utterance)


    dict_rand.append(dict_reference)     
   

###############################################################################
                    # Saving Results as Dictionaries #
###############################################################################
       
filename = 'dict_top' 
numpy.save( path_out + filename , dict_top )

filename = 'dict_low' 
numpy.save(  path_out + filename , dict_low )


filename = 'dict_rand' 
numpy.save(  path_out + filename , dict_rand )







                    
