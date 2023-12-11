from ast import Add
import atexit
from re import T
from tkinter import N, NO
from traceback import print_tb
from turtle import position
import numpy as np
import os
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras import backend as K

from blurpool import BlurPool2D
# from keras_flops import get_flops

from utils import *
from BasicBlock import *


def subtractPixel_layer(val):
    return val + 1 - val * 2

def addInverseMap(input_map, last_output):
    merge_restore = Multiply()([last_output, input_map]) # input : 0-background / 1-points
    inverse_map = Lambda(subtractPixel_layer, name="lambda_layer")(input_map) # restore_output : 0-points position / 1-informative background
    inverse_restore = Multiply()([last_output, inverse_map])
    merge_restore = Add()([merge_restore, inverse_restore])
    merge_restore = BatchNormalization(axis=3)(merge_restore)

    return merge_restore


def SIFTAttNet_v4_encoder(input_size=None,
                  name='SIFTAttNet_v4_encoder',
                  input_channel=2, blocks=6, non_linear=True, 
                  atten = None, activation_name = 'relu', FAM_nodule = False,  droppath_rate=0, lastCBAM=False):
    filters = [16, 32, 64, 128]

    input1 = layers.Input(shape=(None, None, input_channel), name="input1")
    input_map = layers.Input(shape=(None, None, 1), name="input_map")

    map_feature = Multiply()([input1, input_map])
    map_feature = BatchNormalization(axis=3)(map_feature)


    concat_B2_map = DownSamplingModule(map_feature, filters[0], block_num='map_feature_2', position='EN') # /2
    concat_B3_map = DownSamplingModule(concat_B2_map, filters[1], block_num='map_feature_4', position='EN') # /4
    concat_B4_map = DownSamplingModule(concat_B3_map, filters[2], block_num='map_feature_8', position='EN') # /8

    # Encoder
    conv1_layers = EncodeLayer(input1, filter=filters[0], block_num=1, position='EN',activation_name = activation_name, droppath_rate=droppath_rate)
    pool1 = DownSamplingModule(conv1_layers, filters[0], block_num='pool1', position='EN',activation_name = activation_name)

    pool1wMap = Add()([pool1, concat_B2_map])
    pool1wMap = BatchNormalization(axis=3)(pool1wMap)
    if lastCBAM:
        pool1wMap = cbam_block(pool1wMap, 8)
    if FAM_nodule:
        pool1wMap = FAM(input = pool1, pre_conv = pool1wMap, filter = filters[0], position='input1')
    B2 = Conv2D(filters[1], (3, 3), strides=2, padding='same', name='Conv_stride2_B2_EN')(input1)
    conv2_layers = EncodeLayer(pool1wMap, downsample_input=B2, filter=filters[1], block_num=2, position='EN',activation_name = activation_name, droppath_rate=droppath_rate)
    
    pool2 = DownSamplingModule(conv2_layers, filters[1], block_num='pool2', position='EN',activation_name = activation_name)


    pool2wMap = Add()([pool2, concat_B3_map])
    pool2wMap = BatchNormalization(axis=3)(pool2wMap)
    if lastCBAM:
        pool2wMap = cbam_block(pool2wMap, 8)
    if FAM_nodule:
        pool2wMap = FAM(input = pool2, pre_conv = pool2wMap, filter = filters[1], position='input2')
    B3 = Conv2D(filters[2], (3, 3), strides=2, padding='same', name='Conv_stride2_B3_EN')(B2)
    conv3_layers = EncodeLayer(pool2wMap, downsample_input=B3, filter=filters[2], block_num=3, position='EN',activation_name = activation_name, droppath_rate=droppath_rate)
    pool3 = DownSamplingModule(conv3_layers, filters[2], block_num='pool3', position='EN',activation_name = activation_name)


    # Bottleneck
    print('******* Bottleneck *******')
    pool3wMap = Add()([pool3, concat_B4_map])
    pool3wMap = BatchNormalization(axis=3)(pool3wMap)
    if lastCBAM:
        pool3wMap = cbam_block(pool3wMap, 8)
    if FAM_nodule:
        pool3wMap = FAM(input = pool3, pre_conv = pool3wMap, filter = filters[2], position='input3')
    B4 = Conv2D(filters[3], (3, 3), strides=2, padding='same', name='Conv_stride2_B4_EN')(B3)
    print('B4',B4.shape)

    bottleneck = BottleNeck(pool3wMap, B4, filters[3], block_num=4, position='EN', atten =atten, activation_name = activation_name, droppath_rate=droppath_rate)
    print('bottleneck', bottleneck.shape)

    bottleneck = BasicConv(bottleneck, filter=filters[3], block_name=403, transpose=True, position='EN',activation_name = activation_name)

    print('******* Decode *******')
    # Decoder
    db3 = UpSamplingModule(bottleneck, filters[3], block_num='db3', position='NB',activation_name = activation_name)

    db3 = Concatenate(name='concat_db3_NB')([db3, conv3_layers])
    db3 = DecodeLayer(db3, filter=filters[2], block_num='db3', blocks=blocks, position='NB',activation_name = activation_name, droppath_rate=droppath_rate)

    db2 = UpSamplingModule(db3, filters[2], block_num='db2', position='NB',activation_name = activation_name)

    db2 = Concatenate(name='concat_db2_NB')([db2, conv2_layers])
    db2 = DecodeLayer(db2, filter=filters[1], block_num='db2', blocks=blocks, position='NB',activation_name = activation_name, droppath_rate=droppath_rate)

    db1 = UpSamplingModule(db2, filters[1], block_num='db1', position='NB',activation_name = activation_name)

    db1 = Concatenate(name='concat_db1_NB')([db1, conv1_layers])
    db1 = DecodeLayer(db1, filter=filters[0], block_num='db1', blocks=blocks, position='NB',activation_name = activation_name, droppath_rate=droppath_rate)

    conv81 = UpSampling2D(size=(8, 8), name='upSample_8_NB')(bottleneck)
    conv82 = UpSampling2D(size=(4, 4), name='upSample_4_NB')(db3)
    conv83 = UpSampling2D(size=(2, 2), name='upSample_2_NB')(db2)
    conv8 = concatenate([conv81, conv82, conv83, db1], axis=-1)
    conv8 = Conv2D(input_channel, (1, 1))(conv8)

    output=Add()([input1,conv8])
    output = Conv2D(input_channel, (1, 1))(output)
    output = BatchNormalization(axis=3)(output)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[input1, input_map], outputs=output)

    return model


def DualResBlock(image, map, filter, CBAM=False, debug =True):
    conv_concat = Concatenate()([image, map])
    conv3 = Conv2D(filter, kernel_size=(3,3), padding="same")(conv_concat)
    act3 = Activation('relu')(conv3)
    conv4 = Conv2D(filter, kernel_size=(3,3), padding="same")(act3)
    act4 = Activation('relu')(conv4)

    if debug:
        output = Add()([act4, image])
    
    else:
        conv_concat = Concatenate()([image, map])
        conv6 = Conv2D(filter, kernel_size=(3,3), padding="same")(conv_concat)
        output = Add()([act4, conv6])

    
    if CBAM:
        output = cbam_block(output, 8)
        print('CBAM')

    return output


def SIFTAttNet_v4_encoder_CBAM(input_size=None,
                  name='SIFTAttNet_v4_encoder',
                  input_channel=2, blocks=6, non_linear=True, activation_name = 'relu', 
                  var=0.03, pool='DownSamplingModule', upsample = 'UpSamplingModule', dilated = True,
                  atten = None,  FAM_nodule = False, CBAM = False, dualConv = True, debug = True, droppath_rate=0, lastCBAM=False):
    filters = [16, 32, 64, 128]

    input1 = layers.Input(shape=(None, None, input_channel), name="input1")
    input_map = layers.Input(shape=(None, None, 1), name="input_map")

    map_feature = Multiply()([input1, input_map])
    map_feature = BatchNormalization(axis=3)(map_feature)


    concat_B2_map = DownSamplingModule(map_feature, filters[0], block_num='map_feature_2', position='EN') # /2
    concat_B3_map = DownSamplingModule(concat_B2_map, filters[1], block_num='map_feature_4', position='EN') # /4
    concat_B4_map = DownSamplingModule(concat_B3_map, filters[2], block_num='map_feature_8', position='EN') # /8

    # Encoder
    conv1_layers = EncodeLayer(input1, filter=filters[0], block_num=1, position='EN',activation_name = activation_name, 
                               var=var, non_linear=non_linear, droppath_rate=droppath_rate, dilated = dilated)
    
    if pool =="Maxpool":
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_layers)
    elif pool == "Maxblurpool":
        pool1 = BlurPool2D()(conv1_layers)
    elif pool == "DownSamplingModule":
        pool1 = DownSamplingModule(conv1_layers, filters[0], block_num='pool1', position='EN',activation_name = activation_name)

    pool1wMap = Add()([pool1, concat_B2_map])
    pool1wMap = BatchNormalization(axis=3)(pool1wMap)
    if FAM_nodule:
        pool1wMap = FAM(input = pool1, pre_conv = pool1wMap, filter = filters[0], position='input1')
    if dualConv:
        pool1wMap = DualResBlock(image=pool1, map = pool1wMap, filter = filters[0], CBAM=CBAM, debug = debug)
        print('pool1wMap' , pool1wMap.shape)
    B2 = Conv2D(filters[1], (3, 3), strides=2, padding='same', name='Conv_stride2_B2_EN')(input1)
    conv2_layers = EncodeLayer(pool1wMap, downsample_input=B2, filter=filters[1], block_num=2, position='EN',activation_name = activation_name, 
                               var=var, non_linear=non_linear, droppath_rate=droppath_rate, dilated = dilated)
    if lastCBAM:
        conv2_layers = cbam_block(conv2_layers, 8)
    
    if pool =="Maxpool":
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_layers)
    elif pool == "Maxblurpool":
        pool2 = BlurPool2D()(conv2_layers)
    elif pool == "DownSamplingModule":
        pool2 = DownSamplingModule(conv2_layers, filters[1], block_num='pool2', position='EN',activation_name = activation_name)
    

    pool2wMap = Add()([pool2, concat_B3_map])
    pool2wMap = BatchNormalization(axis=3)(pool2wMap)
    if FAM_nodule:
        pool2wMap = FAM(input = pool2, pre_conv = pool2wMap, filter = filters[1], position='input2')
    if dualConv:
        pool2wMap = DualResBlock(image=pool2, map = pool2wMap, filter = filters[1], CBAM=CBAM, debug = debug)
        print('pool2wMap' , pool2wMap.shape)
    B3 = Conv2D(filters[2], (3, 3), strides=2, padding='same', name='Conv_stride2_B3_EN')(B2)
    conv3_layers = EncodeLayer(pool2wMap, downsample_input=B3, filter=filters[2], block_num=3, position='EN',activation_name = activation_name, 
                               var=var, non_linear=non_linear, droppath_rate=droppath_rate, dilated = dilated)
    if lastCBAM:
        conv3_layers = cbam_block(conv3_layers, 8)

    if pool =="Maxpool":
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_layers)
    elif pool == "Maxblurpool":
        pool3 = BlurPool2D()(conv3_layers)
    elif pool == "DownSamplingModule":
        pool3 = DownSamplingModule(conv3_layers, filters[2], block_num='pool3', position='EN',activation_name = activation_name)


    # Bottleneck
    print('******* Bottleneck *******')
    pool3wMap = Add()([pool3, concat_B4_map])
    pool3wMap = BatchNormalization(axis=3)(pool3wMap)
    if FAM_nodule:
        pool3wMap = FAM(input = pool3, pre_conv = pool3wMap, filter = filters[2], position='input3')
    if dualConv:
        pool3wMap = DualResBlock(image=pool3, map = pool3wMap, filter = filters[2], CBAM=CBAM, debug = debug)
        print('pool1wMap' , pool3wMap.shape)

    B4 = Conv2D(filters[3], (3, 3), strides=2, padding='same', name='Conv_stride2_B4_EN')(B3)
    print('B4',B4.shape)

    bottleneck = BottleNeck(pool3wMap, B4, filters[3], block_num=4, position='EN', atten =atten, activation_name = activation_name, 
                            var=var, non_linear=non_linear, droppath_rate=droppath_rate, dilated = dilated)
    print('bottleneck', bottleneck.shape)
    bottleneck = BasicConv(bottleneck, filter=filters[3], block_name=403, transpose=True, position='EN',activation_name = activation_name)
    if lastCBAM:
        bottleneck = cbam_block(bottleneck, 8)

    print('******* Decode *******')
    # Decoder

    if upsample == 'UpSamplingModule':
        db3 = UpSamplingModule(bottleneck, filters[3], block_num='db3', position='NB',activation_name = activation_name)
    else:
        db3 = UpSampling2D(size=(2, 2))(bottleneck)

    db3 = Concatenate(name='concat_db3_NB')([db3, conv3_layers])
    db3 = DecodeLayer(db3, filter=filters[2], block_num='db3', blocks=blocks, position='NB',activation_name = activation_name, var=var, non_linear=non_linear, droppath_rate=droppath_rate)

    if upsample == 'UpSamplingModule':
        db2 = UpSamplingModule(db3, filters[2], block_num='db2', position='NB',activation_name = activation_name)
    else:
        db2 = UpSampling2D(size=(2, 2))(db3)

    db2 = Concatenate(name='concat_db2_NB')([db2, conv2_layers])
    db2 = DecodeLayer(db2, filter=filters[1], block_num='db2', blocks=blocks, position='NB',activation_name = activation_name, var=var, non_linear=non_linear, droppath_rate=droppath_rate)

    if upsample == 'UpSamplingModule':
        db1 = UpSamplingModule(db2, filters[1], block_num='db1', position='NB',activation_name = activation_name)
    else:
        db1 = UpSampling2D(size=(2, 2))(db2)

    db1 = Concatenate(name='concat_db1_NB')([db1, conv1_layers])
    db1 = DecodeLayer(db1, filter=filters[0], block_num='db1', blocks=blocks, position='NB',activation_name = activation_name, var=var, non_linear=non_linear, droppath_rate=droppath_rate)

    conv81 = UpSampling2D(size=(8, 8), name='upSample_8_NB')(bottleneck)
    conv82 = UpSampling2D(size=(4, 4), name='upSample_4_NB')(db3)
    conv83 = UpSampling2D(size=(2, 2), name='upSample_2_NB')(db2)
    conv8 = concatenate([conv81, conv82, conv83, db1], axis=-1)
    conv8 = Conv2D(input_channel, (1, 1))(conv8)

    output=Add()([input1,conv8])
    output = Conv2D(input_channel, (1, 1))(output)
    output = BatchNormalization(axis=3)(output)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[input1, input_map], outputs=output)

    return model


if __name__ == '__main__':
    model = SIFTAttNet_v4_encoder_CBAM(blocks=5, input_channel=2, droppath_rate=0.2, dualConv=True, CBAM=True, lastCBAM=False)
    model.summary()

    inputs = tf.random.uniform((32, 40, 40, 2), maxval=1)
    input_map = tf.random.uniform((32, 40, 40, 1), maxval=1)
    print(inputs.shape)

    output = model.predict([inputs, input_map])
    samples_to_predict = np.array(output)
    print(samples_to_predict.shape)

    dot_img_file = './plotModel/' + SIFTAttNet_v4_encoder_CBAM.__name__.split('.')[0] + '.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    print(SIFTAttNet_v4_encoder_CBAM.__name__)

    # flops = get_flops(model, batch_size=1)
    # print(f"FLOPS: {flops / 10 ** 9:.04} G")
