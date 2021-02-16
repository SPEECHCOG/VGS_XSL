import os
import scipy,scipy.io
import numpy 
import nltk
from gensim.models import KeyedVectors

path_project = '.../VGS_XSL/'

model_name = ''

path_in = ''       
path_out = '' 

file_out = ''

w2vfile = '.../GoogleNews-vectors-negative300.bin'


filename = 'dict_top'
dict_top = numpy.load(path_in + filename + '.npy')

filename = 'dict_low'
dict_low = numpy.load(path_in + filename + '.npy'  ) 

filename = 'dict_rand'
dict_rand = numpy.load(path_in + filename + '.npy' )

###############################################################################
                         # nltk tool for extracting nouns #
###############################################################################    

model = KeyedVectors.load_word2vec_format(w2vfile, binary=True)
 
def string_to_nouns (string):
    words = nltk.word_tokenize(string)
    tok = nltk.pos_tag(words)
    nouns = [tok[i][0].lower() for i in numpy.arange(len(tok)) if tok[i][1] =='NN' or tok[i][1] =='NNS'  
             or tok[i][1] =='VB' or tok[i][1] =='VBD' or tok[i][1] =='VBG' or tok[i][1] =='VBN' or tok[i][1] =='VBP'
             or tok[i][1] =='JJ'or tok[i][1] =='JJR'or tok[i][1] =='JJS']
#    print(tok)
#    print(nouns)
    return nouns

def count_similar_nouns (nouns1,nouns2):
    cnt = 0
    for word in nouns1:
        cnt += nouns2.count(word)
        #print(word)
    return cnt

def w2v_similarity (noun_list_ref,noun_list_can):

    max_similarities = []
    for n_r in noun_list_ref:        
        noun_vec = []
        for n_c in noun_list_can:                        
            sim = model.similarity(n_r,n_c)
            noun_vec.append(sim)            
        max_similarities.append(numpy.max(noun_vec))
    return round(numpy.mean(max_similarities),3)


def w2v_similarity_without (noun_list_ref,noun_list_can):

    max_similarities = []
    for n_r in noun_list_ref:        
        noun_vec = []
        for n_c in noun_list_can:
            if n_c != n_r :
                try:
                    sim = model.similarity(n_r,n_c)
                    noun_vec.append(sim)
                except:
                    pass#print("An exception occurred in nouns" + n_q + " and " + n_r) 
        max_similarities.append(numpy.max(noun_vec))
    return round(numpy.mean(max_similarities),3)
                            

###############################################################################
                        # top similarity results#
###############################################################################
top_similarity1= []
top_similarity2 = []
top_z = []

printcount = 0
for item in dict_top:
    print('...... top similarities............ ' + str(printcount)  )
    string_q = item[0]
    #print(string_q)
    all_s1 = []
    all_s2 = []
    all_z = []
    noun_list_q = string_to_nouns (string_q)   
     
    for string_r in item[1:]:
        try:
            noun_list_r = string_to_nouns (string_r)
            
            s1 =  w2v_similarity_without(noun_list_q,noun_list_r)          
            s2 = w2v_similarity (noun_list_q,noun_list_r)                      
            z = count_similar_nouns (noun_list_q,noun_list_r) 
            
            all_s1.append(s1)
            all_s2.append(s2)
            #all_s3.append(s3)
            all_z.append(z) 
            
        except:
                pass#print("An exception occurred in nouns")
    
    top_similarity1.append(all_s1)
    top_similarity2.append(all_s2)
    #top_similarity3.append(all_s3)
    top_z.append(all_z)
    
    printcount += 1
    
del dict_top  
###############################################################################
                        # lowest similarity results#
###############################################################################



low_similarity1= []
low_similarity2 = []
low_z = []

lowest_counts_same_words = []
printcount = 0
for item in dict_low:
    print('.........lowest similarities......... ' + str(printcount) )
    string_q = item[0]
    #print(string_q)
    all_z = []
    all_s1 = []
    all_s2 = []

    noun_list_q = string_to_nouns (string_q)   
     
    for string_r in item[1:]:
        try:
            noun_list_r = string_to_nouns (string_r)
        
            s1 =  w2v_similarity_without(noun_list_q,noun_list_r)           
            s2 = w2v_similarity (noun_list_q,noun_list_r)
            z = count_similar_nouns (noun_list_q,noun_list_r) 
            
            all_s1.append(s1)
            all_s2.append(s2)
            all_z.append(z) 
            
        except:
                pass#print("An exception occurred in nouns")
    
    low_similarity1.append(all_s1)
    low_similarity2.append(all_s2)
    low_z.append(all_z)
    
    printcount += 1 
    
del dict_low
###############################################################################
                        # random similarity results#
###############################################################################
filename = 'dict_randoms' + '.npy'

random_similarity1= []
random_similarity2= []
random_z = []

printcount = 0
for item in dict_rand:
    print('....... random similarities ........... ' + str(printcount) )
    string_q = item[0]
    #print(string_q)
    all_z = []
    all_s1 = []
    all_s2 = []

    noun_list_q = string_to_nouns (string_q)   
     
    for string_r in item[1:]:
        try:
            noun_list_r = string_to_nouns (string_r)
            
            s1 =  w2v_similarity_without(noun_list_q,noun_list_r)           
            s2 = w2v_similarity (noun_list_q,noun_list_r)
            z = count_similar_nouns (noun_list_q,noun_list_r) 
            
            all_s1.append(s1)
            all_s2.append(s2)
            all_z.append(z) 
            
        except:
                pass#print("An exception occurred in nouns")
    
    random_similarity1.append(all_s1)
    random_similarity2.append(all_s2)    
    random_z.append(all_z)
    
    printcount += 1 
    
del dict_rand
###############################################################################
                        # saving the results #
###############################################################################
scipy.io.savemat(path_out + file_out , {'top_similarity1':top_similarity1,'top_similarity2':top_similarity2,
                                                             'low_similarity1':low_similarity1,'low_similarity2':low_similarity2,
                                                             'random_similarity1':random_similarity1,'random_similarity2':random_similarity2}) 


