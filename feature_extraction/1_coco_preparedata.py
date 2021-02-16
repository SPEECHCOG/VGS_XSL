import os
import numpy as np


images_path = ".../data/MSCOCO/val2014/val2014"
speech_path = ".../data/SPEECH-COCO/val2014/val2014/wav"

path_project = '.../VGS_XSL/'
path_out =  os.path.join(path_project ,'output/step_1/coco/')
name_save = 'data_list.mat'

list_of_images = os.listdir(images_path)
list_of_images.sort()

list_of_wav_files = []
image_ids = []
counter = 0
for folder in list_of_images:
    data_dir = os.path.join(images_path, folder)
    image_id =  data_dir [-16:-4]
    image_id = str(np.int(image_id)) 
    image_ids.append(image_id)
#    print((image_id + '_'))
#    print('...................................................................')
#    print(counter)
    counter += 1
    wav_count_image = 0
    for f_name in os.listdir(speech_path):
        if f_name.startswith(image_id+ '_'):
            list_of_wav_files.append(f_name)
            wav_count_image += 1

    #print(wav_count_image)

import scipy,scipy.io
image_ids = np.array(image_ids, dtype=np.object)
scipy.io.savemat(path_out + name_save,
                 mdict={'list_of_wav_files':list_of_wav_files,'list_of_images':list_of_images,'image_ids':image_ids}) 


   
# print('........................................................................')
# for item in image_ids:
#     print(item+ '_')
