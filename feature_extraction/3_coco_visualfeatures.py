
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

path_images = ".../MSCOCO/val2014/val2014"
path_speech = ".../SPEECH-COCO/val2014/val2014/wav"

path_in =  os.path.join(path_project ,'output/step_1/coco/')
path_out =  os.path.join(path_project ,'output/step_3/coco/')

file_in = 'processed_data_list.mat'
file_out = 'vgg'

visual_data = scipy.io.loadmat(path_in + file_in, variable_names=['list_of_images','image_id_all'])

list_of_images = visual_data['list_of_images']
temp = list_of_images[0]
list_of_images = []
for item in temp:
    item = item[0]
    correct_item = item.strip()
    list_of_images.append(correct_item) 
del temp
############################################################################### VGG model

layer_name = 'fc1'
model = VGG16()
print(model.summary())
model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

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
   
filename = path_out + file_out
outfile = open(filename,'wb')
pickle.dump(vgg_out_all,outfile)
    
