
import pickle
import scipy,scipy.io
import os


import cv2
import scipy,scipy.io
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

###############################################################################
path_project = '.../VGS_XSL/'
path_images = os.path.join(path_project , 'data/places/images/')

path_in =  os.path.join(path_project ,'output/step_1/places/')
path_out =  os.path.join(path_project ,'output/step_3/places/')

file_in = 'meta_shuffling.mat'
###############################################################################
outfilenames = ['vgg_val','vgg_train1','vgg_train2','vgg_train3','vgg_train4','vgg_train5','vgg_train6','vgg_train7','vgg_train8','vgg_test']
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
all_imnames = data['all_imnames']

temp = all_imnames
all_imnames = []

for element_wav in temp:  
    correct_element = element_wav.strip()        
    all_imnames.append(correct_element)  
del temp

############################################################################### VGG model

layer_name = 'fc1'
model = VGG16()
print(model.summary())
model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

###############################################################################
for chunk in range(10):
    inds_chunk = inds_chunks[chunk]
    list_of_images = [all_imnames[item] for item in inds_chunk]
    temp = list_of_images[0]
    list_of_images = []
    for item in temp:
        item = item[0]
        correct_item = item.strip()
        list_of_images.append(correct_item) 
    del temp
    
    ############################################################################### VGG model
    vgg_out_all = []
    count = 0
    for item in list_of_images: 
        print(count)
        count += 1
        
        image_original = cv2.imread(path_images + item)
        image_resized = cv2.resize(image_original,(224,224))
        image_input_vgg = preprocess_input(image_resized.reshape((1, 224, 224, 3)))
        vgg_out = model.predict(image_input_vgg)
        vgg_out_all.append(vgg_out)
       
    filename = path_out + outfilenames[chunk]
    outfile = open(filename,'wb')
    pickle.dump(vgg_out_all,outfile)
        
