import os
import scipy.io
import scipy,scipy.io
import numpy 
from sklearn.utils import shuffle


path_project = '.../VGS_XSL/'

############################################################################### If places dataset
path_in =  os.path.join(path_project , 'output/step_1/places/')
path_out =  os.path.join(path_project ,'output/step_1/places/')

infile = 'train.mat'
outfilename = 'meta_shuffling.mat'
n_val = 10000
n_test = 10000

#.............................................................Reading data file

data = scipy.io.loadmat(path_in + infile, variable_names = ['captions', 'imnames', 'speakerids','wavnames']) 
all_wavnames = data['all_wavnames']


list_of_wav_files = []

for element_wav in all_wavnames:  
    correct_element = element_wav.strip()        
    list_of_wav_files.append(correct_element)  

del all_wavnames 

n_alldata = len(list_of_wav_files)
n_train = n_alldata - (n_val + n_test)


inds_all = numpy.arange(0,n_alldata)
inds_all_shuffled = shuffle(inds_all)

inds_val = inds_all_shuffled[0:n_val]
inds_train = inds_all_shuffled [n_val: n_train + n_val]
inds_test = inds_all_shuffled [n_train + n_val : ]

scipy.io.savemat(path_out + outfilename ,{ 'inds_all': inds_all, 'inds_all_shuffled':inds_all_shuffled, 'inds_val':inds_val,'inds_train':inds_train, 'inds_test':inds_test})

