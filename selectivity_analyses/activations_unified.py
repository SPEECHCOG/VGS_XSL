#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:12:46 2019
Modified on Mon Jan  11 11:01:00 2021

@author: khorrami, orasanen
"""
import tensorflow as tf

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = False
#config.gpu_options.per_process_gpu_memory_fraction=0.6
#sess = tf.Session(config=config)

import hdf5storage
import numpy
from os.path import join as pjoin
import numpy
import pickle
import scipy,scipy.io
import scipy.spatial as ss
import os, sys
from sys import getsizeof

###############################################################################
                           # USER DEFINED PARAMETERS #
###############################################################################

if(len(sys.argv) < 4):
    toprint = """
    Usage: python activations_unified.py <modelname> <trainingcorpus> <targetcorpus> <training_flag>
    where
    <modelname>     : "CNN17", "CNN1", or "RNNres"
    <trainingcorpus>: "coco" or "places"
    <targetcorpus>  : "coco" or "brent"
    <training_flag> : "trained" for trained model, 0 for untrained model
    """
    print(toprint)
    raise SystemExit('Too few arguments provided.')

modelname = sys.argv[1]
trainingcorpusname = sys.argv[2]
targetcorpusname = sys.argv[3]

activation_location = '/Volumes/BackupHD/Khazar_temporal_segmentation/activations/'
mainfold = '/Users/rasaneno/rundata/timeanalysis_Okko/'

saveformat = 'MATLAB' # save activations as 'MATLAB' (.mat) or 'pickle'

if(sys.argv[4] == 'trained'):
    wfile = 'model_weights.h5' #'epoch0_weights.h5'
else:
    wfile = 'epoch0_weights.h5'

if(modelname == 'RNNres' or modelname == 'CNN17'):
    model_path = mainfold + '/output/step_4/' + trainingcorpusname + '/' + modelname + '/v3/'
elif(modelname == 'CNN1'):
    model_path = mainfold + '/output/step_4/' + trainingcorpusname + '/' + modelname + '/v13/'

if(sys.argv[4] == 'trained'):
    out_path = activation_location + trainingcorpusname + '-' + targetcorpusname + '/' + modelname + '/'
else:
    out_path = activation_location + trainingcorpusname + '-' + targetcorpusname + '/' + modelname + '_random/'

print(out_path)

if(targetcorpusname == 'brent'):
    audiofeature_path = mainfold + '/output/step_2/' + targetcorpusname + '/'
elif(targetcorpusname == 'coco'):
    audiofeature_path = mainfold + '/output/step_2/coco/'
    logmel_file = audiofeature_path + 'logmel_val3'

path_to_save_meta = pjoin(out_path , 'meta/')
path_to_save_activations = pjoin(out_path , 'layers/')

if not os.path.exists(path_to_save_meta):
    os.makedirs(path_to_save_meta)

if not os.path.exists(path_to_save_activations):
    os.makedirs(path_to_save_activations)

###############################################################################
                # AUDIO DATA : brent #
###############################################################################
from numpy import matlib

def normalize_logmels (input_logmels):
    output_logmels = []
    for logmel_feature in input_logmels:
        #
        logmel_mean = numpy.mean(logmel_feature,axis = 0)
        logmel_std = numpy.std(logmel_feature,axis = 0)
        #
        logmel_mean = matlib.repmat(logmel_mean,logmel_feature.shape[0],1)
        logmel_std = matlib.repmat(logmel_std,logmel_feature.shape[0],1)
        #
        logmel_norm = numpy.divide((logmel_feature-logmel_mean),logmel_std)
        output_logmels.append(logmel_norm)
        #
    return output_logmels




def prepareX(dict_logmel):
    number_of_audios = numpy.shape(dict_logmel)[0]
    number_of_audio_features = numpy.shape(dict_logmel[0])[1]
    if(trainingcorpusname == 'places'):
        len_of_longest_sequence = 1024
    else:
        len_of_longest_sequence = 512
    X = numpy.zeros((number_of_audios ,len_of_longest_sequence, number_of_audio_features),dtype ='float32')
    for k in numpy.arange(number_of_audios):
       logmel_item = dict_logmel[k]
       logmel_item = logmel_item[0:len_of_longest_sequence]
       X[k,len_of_longest_sequence-len(logmel_item):, :] = logmel_item
    return X


def loadData(targetcorpusname):
    if(targetcorpusname == 'brent'):
        data = scipy.io.loadmat((pjoin(audiofeature_path, 'logmel.mat'))) # load raw
        data_content = data['logmel_all']
        logmels = data_content[0]
        logmels_normalized = normalize_logmels(logmels) # mean norming
        Xdata = prepareX(logmels_normalized)
        del logmels
        del logmels_normalized
    elif(targetcorpusname == 'coco'):
        metadata_dir = '/Users/rasaneno/rundata/timeanalysis_Okko/output/step_2/coco/annotations/'
        metadata_splitting = scipy.io.loadmat(metadata_dir + 'meta_shuffling_val.mat' , variable_names = ['inds_all','inds_all_shuffled','inds_test','inds_val','inds_train'])
        inds_train = metadata_splitting ['inds_train'][0]
        inds_val = metadata_splitting ['inds_val'][0]
        inds_test = metadata_splitting ['inds_test'][0]
        filename = logmel_file
        #Xdata_all = loadXdata(filename)
        infile = open(filename ,'rb')
        logmels = pickle.load(infile)
        infile.close()
        logmels_normalized = normalize_logmels(logmels)
        Xdata_all = prepareX(logmels_normalized)
        Xdata = Xdata_all[inds_test]
        del Xdata_all
        del logmels
        del logmels_normalized
    return Xdata


def mycustomloss(y_true,y_pred):
    margin = 0.2
    Sp = y_pred[0::3]
    Si = y_pred[1::3]
    Sc = y_pred[2::3]
    return K.sum(K.maximum(0.0,(Sc-Sp + margin )) + K.maximum(0.0,(Si-Sp + margin )),  axis=0)


import keras
from keras import backend as K
from keras.models import Model
from keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization
from keras.layers import  MaxPooling1D, Conv1D, LSTM
from keras.layers import Lambda, Activation
from keras.layers.merge import add

#..............................................................................  Residual LSTM

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

###############################################################################
## Model definitions

def create_RNNres_model():
    layer_names = ['layer0','conv1d_1','lstm_1','lstm_2']
    repeatTimes = [0,3,3,3]
    dropout_size = 0.3
    activation_C='relu'
    activation_R='tanh'
    Y_shape = (4096,)
    if(trainingcorpusname == 'places'):
        X_shape = (1024, 40)
        connection_size = 1024
    else:
        X_shape = (512, 40)
        connection_size = 512
    audio_sequence = Input(shape=X_shape)
    forward1 = Conv1D(64,6, strides = 3, padding="same",activation=activation_C)(audio_sequence)
    dr1 = Dropout(0.1)(forward1)
    resLSTM = make_residual_lstm_layers(dr1, rnn_width=512, rnn_depth=3, rnn_dropout=0.0)
    recurrentEnd = resLSTM
    out_audio = Reshape([int(recurrentEnd.shape[1])],name='reshape_audio')(recurrentEnd)
    out_audio = Dense(connection_size,activation='linear',name='dense_audio')(out_audio)
    out_audio = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(out_audio)
    visual_sequence = Input(shape=Y_shape)
    out_visual = Dense(connection_size,activation='linear',name='dense_visual')(visual_sequence) #25
    out_visual = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(out_visual)
    L_layer = keras.layers.dot([out_visual,out_audio],axes=-1,name='dot')
    model = Model(inputs=[visual_sequence, audio_sequence], outputs = L_layer)
    print(model.summary())
    #
    ############################################################################### for using previously trained model
    model.load_weights(pjoin(model_path , wfile))
    #
    ###############################################################################
                            # defining the new Audio model #
    ###############################################################################
    #
    new_audio_model = Model(inputs=audio_sequence,outputs=out_audio)
    #
    for n in range (9):
        new_audio_model.layers[n].set_weights(model.layers[n].get_weights())
    #
    new_audio_model.layers[9].set_weights(model.layers[10].get_weights())
    new_audio_model.layers[10].set_weights(model.layers[12].get_weights())
    new_audio_model.layers[11].set_weights(model.layers[14].get_weights())
    print(new_audio_model.summary())
    return new_audio_model, layer_names, repeatTimes

def create_CNN1_v3_model():
    dropout_size = 0.3
    activation_C='relu'
    activation_R='tanh'
    layer_names = ['layer0','layer1','layer2','layer3','layer5','layer7','layer9']
    repeatTimes = [0,0,0,0,2,4,8]
    if(trainingcorpusname == 'places'):
        audio_sequence = Input(shape=(1024, 40))
        connection_size = 1024
    else:
        audio_sequence = Input(shape=(512, 40))
        connection_size = 512
    # layer 1: 30 ms
    f0 = Conv1D(64,3,padding="same",activation=activation_C,name='layer1')(audio_sequence)       # conv 1
    d0 = Dropout(dropout_size)(f0)
    # layer 2: 50 ms
    f1 = Conv1D(128,3,padding="same",activation=activation_C,name='layer2')(d0)                   # conv 2
    d1 = Dropout(dropout_size)(f1)
    #layer 3: 90 ms
    f2 = Conv1D(128,5,padding="same",activation=activation_C,name='layer3')(d1)                   # Conv 3
    d2 = Dropout(dropout_size)(f2)
    #layer 4: 110 ms
    pool0 = MaxPooling1D(3,strides = 2,padding='same',name='layer4')(d2)                          # maxpool 1
    #layer 5: 150 ms
    f3 = Conv1D(256,3,padding="same",activation=activation_C,name='layer5')(pool0)                # conv 4
    d3 = Dropout(dropout_size)(f3)
    #layer 6: 190 ms
    pool1 = MaxPooling1D(3,strides = 2,padding='same',name='layer6')(d3)                          # maxpool 2
    #layer 7: 270 ms
    f4 = Conv1D(512,3,padding="same",activation=activation_C,name='layer7')(pool1)                # conv 5
    d4 = Dropout(dropout_size)(f4)
    #layer 8: 350 ms
    pool2 = MaxPooling1D(3,strides = 2,padding='same',name='layer8')(d4)                          # maxpool
    #layer 9: 770 ms
    f5 = Conv1D(1024,5,padding="same",activation=activation_C,name='layer9')(pool2)  # conv 6
    d5 = Dropout(dropout_size)(f5)
    #layer 10: maxpooling across entire caption
    pool3 = MaxPooling1D(256,padding='same',name='layer10')(d5)
    #
    out_audio = Reshape([int(pool3.shape[2])],name='reshape_audio')(pool3) #22
    out_audio = Dense(connection_size,activation='linear',name='dense_audio')(out_audio) #24
    out_audio = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(out_audio)
    #.............................................................................. Visual Network
    #
    visual_sequence = Input(shape=(4096,))
    #
    out_visual = Dense(connection_size,activation='linear',name='dense_visual')(visual_sequence) #25
    #
    out_visual = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(out_visual)
    #.............................................................................. combining audio-visual networks
    #
    L_layer = keras.layers.dot([out_visual,out_audio],axes=-1,name='dot')
    #
    model = Model(inputs=[visual_sequence, audio_sequence], outputs = L_layer)
    print(model.summary())
    #
    ############################################################################### for using previously trained model
    model.load_weights(pjoin(model_path , wfile))
    #
    ###############################################################################
                            # defining the new Audio model #
    ###############################################################################
    #
    new_audio_model = Model(inputs=audio_sequence,outputs=out_audio)
    #
    for n in range (17):
        new_audio_model.layers[n].set_weights(model.layers[n].get_weights())
    #
    new_audio_model.layers[17].set_weights(model.layers[18].get_weights())
    new_audio_model.layers[18].set_weights(model.layers[20].get_weights())
    new_audio_model.layers[19].set_weights(model.layers[22].get_weights())
    #
    print(new_audio_model.summary())
    return new_audio_model, layer_names, repeatTimes

def create_CNN1_v13_model():
    layer_names = ['layer0','layer1','layer2','layer3','layer5','layer7','layer9']
    repeatTimes = [0,0,0,0,2,4,8]
    dropout_size = 0.3
    activation_C='relu'
    activation_R='tanh'
    if(trainingcorpusname == 'places'):
        audio_sequence = Input(shape=(1024, 40))
        connection_size = 1024
    else:
        audio_sequence = Input(shape=(512, 40))
        connection_size = 512
    # layer 1: 30 ms
    f0 = Conv1D(512,3,padding="same",activation=activation_C,name='layer1')(audio_sequence)       # conv 1
    d0 = Dropout(dropout_size)(f0)
    # layer 2: 50 ms
    f1 = Conv1D(512,3,padding="same",activation=activation_C,name='layer2')(d0)                   # conv 2
    d1 = Dropout(dropout_size)(f1)
    #layer 3: 90 ms
    f2 = Conv1D(512,5,padding="same",activation=activation_C,name='layer3')(d1)                   # Conv 3
    d2 = Dropout(dropout_size)(f2)
    #layer 4: 110 ms
    pool0 = MaxPooling1D(3,strides = 2,padding='same',name='layer4')(d2)                          # maxpool 1
    #layer 5: 150 ms
    f3 = Conv1D(512,3,padding="same",activation=activation_C,name='layer5')(pool0)                # conv 4
    d3 = Dropout(dropout_size)(f3)
    #layer 6: 190 ms
    pool1 = MaxPooling1D(3,strides = 2,padding='same',name='layer6')(d3)                          # maxpool 2
    #layer 7: 270 ms
    f4 = Conv1D(512,3,padding="same",activation=activation_C,name='layer7')(pool1)                # conv 5
    d4 = Dropout(dropout_size)(f4)
    #layer 8: 350 ms
    pool2 = MaxPooling1D(3,strides = 2,padding='same',name='layer8')(d4)                          # maxpool 3
    #layer 9: 770 ms
    f5 = Conv1D(512,5,padding="same",activation=activation_C,name='layer9')(pool2)  # conv 6
    d5 = Dropout(dropout_size)(f5)
    #layer 10: maxpooling across entire caption
    pool3 = MaxPooling1D(512,padding='same',name='layer10')(d5)
    out_audio = Reshape([int(pool3.shape[2])],name='reshape_audio')(pool3) #22
    out_audio = Dense(connection_size,activation='linear',name='dense_audio')(out_audio) #24
    out_audio = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(out_audio)
    visual_sequence = Input(shape=(4096,))
    out_visual = Dense(connection_size,activation='linear',name='dense_visual')(visual_sequence) #25
    out_visual = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(out_visual)
    L_layer = keras.layers.dot([out_visual,out_audio],axes=-1,name='dot')
    model = Model(inputs=[visual_sequence, audio_sequence], outputs = L_layer)
    print(model.summary())
    #
    ############################################################################### for using previously trained model
    model.load_weights(pjoin(model_path , wfile))
    ###############################################################################
                            # defining the new Audio model #
    ###############################################################################
    new_audio_model = Model(inputs=audio_sequence,outputs=out_audio)
    for n in range (17):
        new_audio_model.layers[n].set_weights(model.layers[n].get_weights())
    #
    new_audio_model.layers[17].set_weights(model.layers[18].get_weights())
    new_audio_model.layers[18].set_weights(model.layers[20].get_weights())
    new_audio_model.layers[19].set_weights(model.layers[22].get_weights())
    print(new_audio_model.summary())
    return new_audio_model, layer_names, repeatTimes

def create_CNN17_model():
    layer_names = ['layer0','conv1d_1','conv1d_2','conv1d_3','conv1d_4','conv1d_5']
    repeatTimes = [0,0,0,2,4,8]
    dropout_size = 0.3
    activation_C='relu'
    activation_R='tanh'
    #.............................................................................. Audio Network
    if(trainingcorpusname == 'places'):
        audio_sequence = Input(shape=(1024, 40))
        connection_size = 1024
    else:
        audio_sequence = Input(shape=(512, 40))
        connection_size = 512
    forward0 = Conv1D(128,1,padding="same",activation=activation_C)(audio_sequence)
    dr0 = Dropout(dropout_size)(forward0)
    forward1 = Conv1D(256,11,padding="same",activation=activation_C)(dr0)
    dr1 = Dropout(dropout_size)(forward1)
    pool1 = MaxPooling1D(3,strides = 2, padding='same')(dr1)
    forward2 = Conv1D(512,17,padding="same",activation=activation_C)(pool1)
    dr2 = Dropout(dropout_size)(forward2)
    pool2 = MaxPooling1D(3,strides = 2,padding='same')(dr2)
    forward3 = Conv1D(512,17,padding="same",activation=activation_C)(pool2)
    dr3 = Dropout(dropout_size)(forward3)
    pool3 = MaxPooling1D(3,strides = 2,padding='same')(dr3) # maxpooling across entire caption
    if(trainingcorpusname == 'places'):
        forward4 = Conv1D(1024,17,padding="same",activation=activation_C)(pool3)
    else:
        forward4 = Conv1D(512,17,padding="same",activation=activation_C)(pool3)
    dr4 = Dropout(dropout_size)(forward4)
    pool4 = MaxPooling1D(128,padding='same')(dr4) # maxpooling across entire caption
    out_audio = Reshape([int(dr4.shape[2])],name='reshape_audio')(pool4)
    out_audio = Dense(connection_size,activation='linear',name='dense_audio')(out_audio)
    out_audio = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(out_audio)
    #.............................................................................. Visual Network
    visual_sequence = Input(shape=(4096,))
    out_visual = Dense(connection_size,activation='linear',name='dense_visual')(visual_sequence) #25
    out_visual = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(out_visual)
    #.............................................................................. combining audio-visual networks
    L_layer = keras.layers.dot([out_visual,out_audio],axes=-1,name='dot')
    model = Model(inputs=[visual_sequence, audio_sequence], outputs = L_layer)
    print(model.summary())
    #
    ############################################################################### for using previously trained model
    model.load_weights(pjoin(model_path , wfile))
    #
    ###############################################################################
                            # defining the new Audio model #
    ###############################################################################
    new_audio_model = Model(inputs=audio_sequence,outputs=out_audio)
    #
    for n in range (15):
        new_audio_model.layers[n].set_weights(model.layers[n].get_weights())
    #
    new_audio_model.layers[15].set_weights(model.layers[16].get_weights())
    new_audio_model.layers[16].set_weights(model.layers[18].get_weights())
    new_audio_model.layers[17].set_weights(model.layers[20].get_weights())
    print(new_audio_model.summary())
    return new_audio_model, layer_names, repeatTimes

###############################################################################
                    # Activation of hidden layers #
###############################################################################

Xdata = loadData(targetcorpusname)

number_of_audios = Xdata.shape[0]
len_of_longest_sequence = Xdata.shape[1]
number_of_audio_features = Xdata.shape[2]

###############################################################################
                    # meta data to be saved once #
###############################################################################
n_examples = number_of_audios

if(modelname == 'CNN1'):
    [model, layer_names, repeatTimes] = create_CNN1_v13_model()
elif(modelname == 'CNN17'):
    [model, layer_names, repeatTimes] = create_CNN17_model()
elif(modelname == 'RNNres'):
    [model, layer_names, repeatTimes] = create_RNNres_model()

data = Xdata
xx = 0

print(data.shape)

for layer_name in layer_names:
    print('............... ' + layer_name + ' ..............')
    if layer_name == 'layer0':
        layer_output = data
    else:
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        layer_output = intermediate_layer_model.predict(data)
    name_to_save = layer_name
    name_to_save = 'layer_' + str(xx)
    filename = pjoin(path_to_save_activations , name_to_save)
    if(saveformat == 'pickle'):
        outfile = open(filename,'wb')
        filters = layer_output
        pickle.dump(filters,outfile)
        outfile.close()
    elif(saveformat == 'MATLAB'):
        if(2**31-getsizeof(layer_output) > 0): # faster save with v5 format if smaller than 2 GB
            scipy.io.savemat(filename + '.mat',{'filters':layer_output})
        else: # slower save with v7.3 format if larger than 2 GB
            hdf5storage.savemat(filename + '.mat',{'filters':layer_output})
    else:
        raise SystemExit('Output format not defined. Must be pickle or MATLAB.')
    xx = xx+1

################################################################################
#saving the name and information of the used layers
numpy.save(path_to_save_meta + 'layer_names' , layer_names)
numpy.save(path_to_save_meta + 'number_of_examples' , n_examples)
numpy.save(path_to_save_meta + 'repeatTimes' , repeatTimes)

###############################################################################
