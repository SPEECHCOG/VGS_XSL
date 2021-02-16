import os
import scipy.io
import scipy,scipy.io
import numpy 
import nltk
from gensim.models import KeyedVectors

path_project = '.../VGS_XSL/'

path_in = os.path.join(path_project ,'output/step_1/places/')
file_in = 'places_testset_captions.mat'

path_out =  os.path.join(path_project ,'')
file_out = 'places_testset_spellcheck.mat'

w2vfile = '.../GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(w2vfile, binary=True) 


def extract_captions (target):
    unit_list = []
    for utterance in target:
        utterance = (utterance.strip())  
        unit_list.append((utterance))    
    return  unit_list

def string_to_nouns (string):
    words = nltk.word_tokenize(string)
    tok = nltk.pos_tag(words)
    nouns = [tok[i][0].lower() for i in numpy.arange(len(tok)) if tok[i][1] =='NN' or tok[i][1] =='NNS'  
             or tok[i][1] =='VB' or tok[i][1] =='VBD' or tok[i][1] =='VBG' or tok[i][1] =='VBN' or tok[i][1] =='VBP'
             or tok[i][1] =='JJ'or tok[i][1] =='JJR'or tok[i][1] =='JJS']
    return nouns

metadata_file = path_in + file_in
data = scipy.io.loadmat(metadata_file)  
captions = data['captions']
caption_list = extract_captions(captions)
###############################################################################
                        #spell check for query words#
###############################################################################
                        
ind_correct = []                        
for counter_valdata,caption in enumerate(caption_list):
    string_q = caption
    noun_list_q = string_to_nouns (string_q)
    print(counter_valdata)
    try:
        model.most_similar(noun_list_q)        
        ind_correct.append(counter_valdata)
    except:        
        pass
        print("An exception occurred in utterance:  " + string_q)
    
###############################################################################
                        # saving the results #
###############################################################################
import scipy
scipy.io.savemat(path_out + file_out, {'ind_correct':ind_correct}) 