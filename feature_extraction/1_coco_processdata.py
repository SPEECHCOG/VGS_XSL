import os
import scipy.io
import scipy,scipy.io
import numpy 



path_project = '.../VGS_XSL/'

path_in = os.path.join(path_project , 'output/step_1/coco/')
file_in = 'data_list.mat'

path_out = os.path.join(path_project ,'output/step_1/coco/')
file_out = 'processed_data_list.mat'

data = scipy.io.loadmat(path_in + file_in,
                 variable_names=['list_of_images','image_ids','list_of_wav_files'])

data_images = data['list_of_images']
data_image_ids = data['image_ids'][0]
data_wav = data['list_of_wav_files']

############################################################################### Pre-Processing data
temp = data_images
data_images = []
for item in temp:
    correct_item = item.strip()
    data_images.append(correct_item) 
############################################################################### MAIN LOOP


list_of_wav_files = []
list_of_wav_counts = []
image_id_all = []
   
for image_counter,image_id in enumerate(data_image_ids):
#    print(image_counter)
#    print('...'+ image_id[0] + '...')
    
    image_id = image_id[0]
    image_id_all.append(image_id)
    wav_of_image = []
    wav_count_image = 0
    
    
    for wav_file in data_wav:
        if wav_file.startswith(image_id+ '_'):
            wav_of_image.append(wav_file)
            wav_count_image += 1

    list_of_wav_files.append(wav_of_image)
    list_of_wav_counts.append(wav_count_image)


image_id_all = numpy.array(image_id_all, dtype=numpy.object)
data_images = numpy.array(data_images, dtype=numpy.object)
list_of_wav_files = numpy.array(list_of_wav_files, dtype=numpy.object)
scipy.io.savemat(path_out + file_out,
                 {'list_of_wav_files':list_of_wav_files,'list_of_wav_counts':list_of_wav_counts,
                  'list_of_images':data_images,'image_id_all':image_id_all})    
