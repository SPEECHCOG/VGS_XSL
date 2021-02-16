import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.Session(config=config) 

###############################################################################
import numpy 
import pickle
import scipy,scipy.io
import scipy.spatial as ss

import os
import scipy,scipy.io
from sklearn.utils import shuffle

###############################################################################
path_project = '.../VGS_XSL/'

path_in_logmel =  os.path.join(path_project , 'output/step_2/places/')
path_in_vgg =  os.path.join(path_project , 'output/step_3/places/')
path_out =  os.path.join(path_project ,'output/step_4/places/')

pretrained_dir = path_out



############################################################################### 
                        # Model #
###############################################################################
import keras
from keras import backend as K
from keras.models import Model

#..............................................................................
from keras.layers import  Input, Reshape, Dense, Dropout
from keras.layers import  Conv1D, LSTM
from keras.layers import Lambda

dropout_size = 0.3
connection_size = 1024
activation_C='relu'
activation_R='tanh'


Y_shape = (4096,)
X_shape = (1024, 40)

#..............................................................................  Residual LSTM
from keras.layers.merge import add

def make_residual_lstm_layers(Myinput, rnn_width, rnn_depth, rnn_dropout):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """
    x = Myinput
    for i in range(rnn_depth):
        return_sequences = i < rnn_depth - 1
        x_rnn = LSTM(rnn_width, activation='tanh', recurrent_dropout=rnn_dropout, dropout=0.1, return_sequences=return_sequences)(x)#(Recurrent 1)
        print(x_rnn.shape)
        if return_sequences:
            # Intermediate layers return sequences, input is also a sequence.
            if i > 0 or Myinput.shape[-1] == rnn_width:
                x = add([x, x_rnn])
            else:
                # Note that the input size and RNN output has to match, due to the sum operation.
                # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
                x = x_rnn
        else:
            # Last layer does not return sequences, just the last element
            # so we select only the last element of the previous output.
            def slice_last(x):
                return x[..., -1, :]
            x = add([Lambda(slice_last)(x), x_rnn]) # recurren 1 + input
    return x
#..............................................................................  RNN
#Chrupala:
#https://github.com/gchrupala/visually-grounded-speech/blob/master/experiments/coco-speech/run.py
#.............................................................................. Audio Network

audio_sequence = Input(shape=X_shape)

# layer 1: 60 ms 
forward1 = Conv1D(64,6, strides = 3, padding="same",activation=activation_C)(audio_sequence)
dr1 = Dropout(0.1)(forward1)


resLSTM = make_residual_lstm_layers(dr1, rnn_width=512, rnn_depth=3, rnn_dropout=0.0)

recurrentEnd = resLSTM
out_audio = Reshape([int(recurrentEnd.shape[1])],name='reshape_audio')(recurrentEnd)
out_audio = Dense(connection_size,activation='linear',name='dense_audio')(out_audio)
out_audio = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(out_audio)

#.............................................................................. Visual Network

visual_sequence = Input(shape=Y_shape) 

out_visual = Dense(connection_size,activation='linear',name='dense_visual')(visual_sequence)
out_visual = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(out_visual)

#.............................................................................. combining audio-visual networks

L_layer = keras.layers.dot([out_visual,out_audio],axes=-1,name='dot')


###############################################################################

model = Model(inputs=[visual_sequence, audio_sequence], outputs = L_layer)
print(model.summary())
model.save('%smodel' % path_out)

############################################################################### for using previously trained model
#model.load_weights(pretrained_dir + 'model_weights.h5')
###############################################################################
                      # Custom loss function #
###############################################################################
                      
def mycustomloss(y_true,y_pred):
    
    margin = 0.2 

    Sp = y_pred[0::3]

    Si = y_pred[1::3]

    Sc = y_pred[2::3]
    
    return K.sum(K.maximum(0.0,(Sc-Sp + margin )) + K.maximum(0.0,(Si-Sp + margin )),  axis=0)                      
                      
                      
###############################################################################
model.compile(loss=mycustomloss, optimizer=keras.optimizers.Adam(lr=1e-04))

############################################################################### Binary target
def make_bin_target (n_sample):
    target = []

    for group_number in range(n_sample):    
        target.append(1)
        target.append(0)
        target.append(0)
        
    return target

   
###############################################################################
def randOrder(n_t):

    random_order = numpy.random.permutation(int(n_t))
    random_order_X = numpy.random.permutation(int(n_t))
    random_order_Y = numpy.random.permutation(int(n_t))
    
    data_orderX = []
    data_orderY = []
      
    for group_number in random_order:
        
        data_orderX.append(group_number)
        data_orderY.append(group_number)
        
        data_orderX.append(group_number)
        data_orderY.append(random_order_Y[group_number])
        
        data_orderX.append(random_order_X[group_number])
        data_orderY.append(group_number)
        
    return data_orderX,data_orderY
###############################################################################
def loadXdata (filename):
    infile = open(filename ,'rb')
    logmel = pickle.load(infile)
    infile.close()
    Xdata = preparX (logmel)
    del logmel
    return Xdata
    
def loadYdata (filename):
    infile = open(filename ,'rb')
    vgg = pickle.load(infile)
    infile.close()
    Ydata = preparY(vgg)
    del vgg 
    return Ydata
############################################################################### 
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

def preparY (dict_vgg):
    Y = numpy.array(dict_vgg)    
    Y = numpy.reshape(Y, [numpy.shape(Y)[0], -1])
    return Y
###############################################################################
def calculate_recallat10( audio_embedd,visual_embedd, sampling_times,  number_of_all_audios, pool):
    
    recall_all = []
    recallat = 10
    
    for trial_number in range(sampling_times):
        
        data_ind = numpy.random.randint(0, high=number_of_all_audios, size=poolsize)
        
        a_embedd = [audio_embedd[item] for item in data_ind]
        v_embedd = [visual_embedd[item] for item in data_ind]
            
        distance_utterance = ss.distance.cdist( a_embedd , v_embedd ,  'cosine') # 1-cosine
        
        r = 0
        for n in range(poolsize):
            #print('###############################################################.....' + str(n))
            ind_audio = n #random.randrange(0,number_of_audios)
                    
            distance_utterance_n = distance_utterance[n] 
            
            sort_index = numpy.argsort(distance_utterance_n)[0:recallat]
            r += numpy.sum((sort_index==ind_audio)*1)
    
        recall_all.append(r)
        del distance_utterance
        
    return recall_all
###############################################################################
                        # defining the new Audio model #
###############################################################################

new_audio_model = Model(inputs=audio_sequence,outputs=out_audio)

for n in range (15):
    new_audio_model.layers[n].set_weights(model.layers[n].get_weights())
   
new_audio_model.layers[15].set_weights(model.layers[16].get_weights())
new_audio_model.layers[16].set_weights(model.layers[18].get_weights())
new_audio_model.layers[17].set_weights(model.layers[20].get_weights())

print(new_audio_model.summary()) 

###############################################################################
                        # defining the new Visual model #
###############################################################################
new_visual_model = Model(inputs=visual_sequence,outputs=out_visual)

new_visual_model.layers[0].set_weights(model.layers[15].get_weights()) # input layer
new_visual_model.layers[1].set_weights(model.layers[17].get_weights())
new_visual_model.layers[2].set_weights(model.layers[19].get_weights())


print(new_visual_model.summary()) 

##############################################################################

############################################################################### 
 
allepochs_valloss = [] 
allepochs_trainloss = [] 
allavRecalls = []
recall_indicator = 0
val_indicator = 1000 
 
###############################################################################
                        # INITIAL VALIDATION #
###############################################################################

############################################################################### loading validation data

filename = path_in_logmel + 'logmel_val' 
Xdata = loadXdata(filename)

filename = path_in_vgg + 'vgg_val' 
Ydata = loadYdata(filename)

number_of_audios = Xdata.shape[0]
len_of_longest_sequence = Xdata.shape[1]
number_of_audio_features = Xdata.shape[2]

number_of_images = Ydata.shape[0]
number_of_visual_features = Ydata.shape[1]
############################################################################### # Finding validation loss #
                        
n_val = Ydata.shape[0]
valorderX,valorderY = randOrder(n_val)
bin_val_triplet = numpy.array(make_bin_target(n_val))

val_primary = model.evaluate( [Ydata[valorderY],Xdata[valorderX] ],bin_val_triplet,batch_size=120) 
print('......................... val primary ... = ' + str(val_primary))
 
allepochs_valloss.append(val_primary)
model.save_weights(path_out + 'epoch0'  + '_weights.h5')

################################################################################ Finding Recall #

audio_embeddings = new_audio_model.predict(Xdata) 
visual_embeddings = new_visual_model.predict(Ydata)
poolsize =  1000
recall_vec = calculate_recallat10( audio_embeddings,visual_embeddings, 10,  number_of_audios , poolsize )

recall10 = numpy.mean(recall_vec)/(poolsize)
print('###############################################################...recall@10 is = ' + str(recall10) )       
allavRecalls.append(recall10) 
    
################################################################################ deleting validation data    
del Ydata
del Xdata
del audio_embeddings
del visual_embeddings

###############################################################################
                        # TRAIN LOOP #
###############################################################################

n_chunks = 8
#.......................................................................... 

for epoch in range(300):
    
    #..........................................................................
    loss_train_chunks = []  
    indx_chunks = shuffle([q+1 for q in range(n_chunks)]) 
    for chunk in indx_chunks: 

        print('......................... epoch .............................' + str(epoch ) )
        print('......................... chunk train .................................' + str(chunk))           
        
        #......................................................................  data loading
        
        filename = path_in_logmel + 'logmel_train' + str(chunk)
        Xdata = loadXdata(filename)
        
        filename = path_in_vgg + 'vgg_train' + str(chunk)
        Ydata = loadYdata(filename)
        
        #...................................................................... data prepration
        n_item = Ydata.shape[0]
        randon_permutation = numpy.random.permutation(int(n_item))
        Ydata = Ydata[randon_permutation]
        Xdata = Xdata[randon_permutation]
        trainorderX,trainorderY = randOrder(Ydata.shape[0])
        bin_train_triplet = numpy.array(make_bin_target(Ydata.shape[0]))          
       
        
        #...................................................................... Fitting 
        
        history = model.fit([Ydata[trainorderY], Xdata[trainorderX] ] , bin_train_triplet, shuffle=False, epochs=1,batch_size=120)  
        loss_train_chunks.append(history.history['loss'][0])
        

        del history                        
        del Xdata
        del Ydata
    
    model.save_weights('%smodel_weights_last.h5' % path_out)
    allepochs_trainloss.append(numpy.mean(loss_train_chunks))
    ########################################################################### VALIDATION
    # .....................................loading validation data

    filename = path_in_logmel + 'logmel_val' 
    Xdata = loadXdata(filename)
    
    filename = path_in_vgg + 'vgg_val' 
    Ydata = loadYdata(filename)
    
    val_epoch = model.evaluate( [Ydata[valorderY],Xdata[valorderX] ],bin_val_triplet,batch_size=120)     
    allepochs_valloss.append(val_epoch)
    print('.................................................................vall epoch is = ' + str(val_epoch) ) 

    ###############################################################################
                                    #  Recall #
    ###############################################################################
    audio_embeddings = new_audio_model.predict(Xdata) 
    visual_embeddings = new_visual_model.predict(Ydata)
    poolsize =  1000
    recall_vec = calculate_recallat10( audio_embeddings,visual_embeddings, 500,  number_of_audios , poolsize )

    recall10 = numpy.mean(recall_vec)/(poolsize)
    print('.................................................................recall@10 is = ' + str(recall10) )       
    allavRecalls.append(recall10)   
                 
    ###############################################################################
    del Ydata
    del Xdata
    del audio_embeddings
    del visual_embeddings
    ###############################################################################
    ############################################################################### saving the best model
 
    if recall10 >= recall_indicator: 
        recall_indicator = recall10
        weights = model.get_weights()
        model.set_weights(weights)
        model.save_weights('%smodel_weights.h5' % path_out)
        
    
    if (epoch+1) %10 == 0:
        model.save_weights(path_out + 'epoch' + str(epoch+1) + '_weights.h5')  
    

    scipy.io.savemat(path_out + 'valtrainloss.mat', 
                 {'allepochs_valloss':allepochs_valloss,'allepochs_trainloss':allepochs_trainloss, 'allavRecalls':allavRecalls})
    
    
