from ast import Add
from re import A, T
from tabnanny import verbose
from tkinter import N, NO, SE
from traceback import print_tb

import numpy as np
import os
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
# import keras
from tensorflow.keras import backend as K
from blurpool import BlurPool2D
from utils import *

'''
Infrastructure
'''

def BasicConv(input_layer, filter=16, dropout=True, kernel_size=(3, 3), batch_norm=False, block_name=1, transpose=False, activation_name = 'relu',
              position='NB'):
    '''
    conv - BN - Activation - Dropout
    '''
    if transpose:
        conv = Conv2DTranspose(filter, kernel_size=kernel_size, padding="same",
                               name="BasicTransConv_{}_{}".format(block_name, position))(input_layer)
    else:
        conv = Conv2D(filter, kernel_size=kernel_size, padding="same",
                      name="BasicConv_{}_{}".format(block_name, position))(input_layer)

    if batch_norm is True:
        conv = BatchNormalization(axis=3, name=f'BN_{block_name}_{position}')(conv)

    conv = Activation(activation_name, name=f'ReLU_{block_name}_{position}')(conv)

    if dropout is True:
        conv = Dropout(0.2, name=f'Dropout_{block_name}_{position}')(conv)

    return conv


def ResBlockScale(input_layer, filter=16, kernel_size=(3, 3), batch_norm=False, block_name=1, scale=3, 
                  activation_name = 'relu', dropout =False, droppath_rate=0,
                  block_type='Plain',  position='NB', name='ResBlock'):
    '''
    Plain Residual:
        conv - (BN) - Activation - conv - (BN) - Activation 
                                            - shortcut  - (BN) - Activation
    Transform Residual:
        LN  -   conv    -   GELU    -   conv 
                                - shortcut  - (BN) - Activation
    Transform_CA Residual:
        LN  -   conv    -   GELU    -   CA    -   conv 
                                - shortcut  - (BN) - Activation
    '''
    str_block_num = str(block_name)

    # if branch == 'b':
    #     activation_name = 'sigmoid'
    # else:
    #     activation_name = 'relu'
    # print("activation_name", activation_name)

    if block_type == 'Plain':
        conv = Conv2D(filter, kernel_size=kernel_size, padding="same",
                      name="{}{}_a_{}_{}".format(block_type, name, str_block_num, position))(input_layer)
        if batch_norm is True:
            conv = BatchNormalization(axis=3, name="{}{}_BN1_{}_{}".format(block_type, name, str_block_num, position))(
                conv)
        conv = Activation(activation_name,
                          name="{}{}_{}1_{}_{}".format(block_type, name, activation_name, str_block_num, position))(
            conv)

        conv = Conv2D(filter, kernel_size=kernel_size, padding="same",
                      name="{}{}_b_{}_{}".format(block_type, name, str_block_num, position))(conv)
        if batch_norm is True:
            conv = BatchNormalization(axis=3, name="{}{}_BN2_{}_{}".format(block_type, name, str_block_num, position))(
                conv)

        conv = Activation(activation_name,
                          name="{}{}_{}2_{}_{}".format(block_type, name, activation_name, str_block_num, position))(
            conv)
        if dropout:
            conv = Dropout(0.2)(conv)

        if droppath_rate > 0:
            conv = StochasticDepth(drop_path_rate=droppath_rate)(conv)
            print('droppath')

        shortcut = tf.keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                                          output_shape=K.int_shape(conv)[1:],
                                          arguments={'scale': scale},
                                          name="Shortcut_{}_{}_{}".format(scale, str_block_num, position))(
            [input_layer, conv])

        res_path = Activation(activation_name,
                              name="{}{}_{}Shortcut_{}_{}".format(block_type, name, activation_name, str_block_num,
                                                                  position))(shortcut)


    elif block_type == 'Transform':

        layernorm = LayerNormalization(axis=3)(input_layer)
        conv = Conv2D(filter, kernel_size=kernel_size, padding="same",
                      name="Trans_ResBlock_a_{}".format(str_block_num))(layernorm)
        conv = tf.keras.activations.gelu(conv)

        conv = Conv2D(filter, kernel_size=kernel_size, padding="same",
                      name="Trans_ResBlock_b_{}".format(str_block_num))(conv)
        conv = tf.keras.activations.gelu(conv)

        shortcut = tf.keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                                          output_shape=K.int_shape(conv)[1:],
                                          arguments={'scale': scale},
                                          name="Shortcut_{}_{}".format(scale, str_block_num))([input_layer, conv])
        res_path = tf.keras.activations.gelu(conv)
        # print('res_path', res_path.shape)

    elif block_type == 'Transform_CA':

        layernorm = LayerNormalization(axis=3)(input_layer)
        conv = Conv2D(filter, kernel_size=kernel_size, padding="same", name="Trans_a_ResBlock{}".format(str_block_num))(
            layernorm)
        conv = tf.keras.activations.gelu(conv)

        CA = CAModule(conv, filter)
        conv = Multiply()([conv, CA])

        conv = Conv2D(filter, kernel_size=kernel_size, padding="same", name="Trans_b_ResBlock{}".format(str_block_num))(
            conv)
        conv = tf.keras.activations.gelu(conv)
        conv = Dropout(0.2)(conv)

        shortcut = tf.keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                                          output_shape=K.int_shape(conv)[1:],
                                          arguments={'scale': scale},
                                          name="Shortcut_{}_{}".format(scale, str_block_num))([input_layer, conv])
        res_path = tf.keras.activations.gelu(conv)
        # print('res_path', res_path.shape)

    return res_path


def CAModule(input_layer, filter, block_num=1, name='CAModule', position='EN'):
    pool = GlobalAveragePooling2D(name="{}_AvgPool_{}_{}".format(name, block_num, position))(input_layer)
    pool = tf.reshape(pool, [-1, 1, 1, filter], name="{}_reshape_{}_{}".format(name, block_num, position))

    conv = Conv2D(filter, (1, 1), padding='same', name="{}_Conv1_{}_{}".format(name, block_num, position))(pool)
    conv = Activation("relu", name="{}_ReLU_{}_{}".format(name, block_num, position))(conv)
    conv = Conv2D(filter, (1, 1), padding='same', name="{}_Conv2_{}_{}".format(name, block_num, position))(conv)
    output = Activation("sigmoid", name="{}_Sigmoid_{}_{}".format(name, block_num, position))(conv)

    return output


def SCM(input_layer, filter=16, block_num: str = 1, position='EN', dropout=False, activation_name= 'relu'):
    str_block_num = block_num

    conv = Conv2D(filter, (3, 3), strides=(1, 1), padding="same",
                  name='SCM_Conv1_{}_{}'.format(str_block_num, position))(input_layer)  # 3x3
    conv = Activation(activation_name, name='SCM_ReLU1_{}_{}'.format(str_block_num, position))(conv)
    conv = Conv2D(filter, (1, 1), strides=(1, 1), padding="same",
                  name='SCM_Conv2_{}_{}'.format(str_block_num, position))(conv)  # 1x1
    conv = Activation(activation_name, name='SCM_ReLU2_{}_{}'.format(str_block_num, position))(conv)

    if dropout:
        conv = Dropout(0.2)(conv)

    conv = Conv2D(filter, (3, 3), strides=(1, 1), padding="same",
                  name='SCM_Conv3_{}_{}'.format(str_block_num, position))(conv)  # 3x3
    conv = Activation(activation_name, name='SCM_ReLU3_{}_{}'.format(str_block_num, position))(conv)
    conv = Conv2D(filter, (1, 1), strides=(1, 1), padding="same",
                  name='SCM_Conv4_{}_{}'.format(str_block_num, position))(conv)  # 1x1
    conv = Activation(activation_name, name='SCM_ReLU4_{}_{}'.format(str_block_num, position))(conv)

    concat = Concatenate(name='SCM_concat_{}_{}'.format(str_block_num, position))([input_layer, conv])
    conv_last = Conv2D(filter, (1, 1), strides=(1, 1), padding="same",
                       name='SCM_last_{}_{}'.format(str_block_num, position))(concat)  # 1x1

    return conv_last


def FAM(input, pre_conv, filter=32, block_num=1, name="FAM", position='EN'):
    """
    input : SCMout
    pre_conv : EB out (follow convolution)
    """
    str_block_num = str(block_num)

    multiplied = Multiply(name="{}_Multiply_{}_{}".format(name, block_num, position))([input, pre_conv])
    multiplied = Conv2D(filter, (3, 3), padding='same', name="{}_Conv1_{}_{}".format(name, block_num, position))(
        multiplied)
    output = Add(name="{}_Add_{}_{}".format(name, block_num, position))([multiplied, pre_conv])

    output = Conv2D(filter, (1, 1), padding='same', name="{}_Conv2_{}_{}".format(name, block_num, position))(
        output)  # debug
    output = BatchNormalization(name="{}_BN_{}_{}".format(name, block_num, position))(output)  # debug

    # output = Activation("relu")(output)

    return output


'''
    Dilated Blocks
'''


def DilatedBlock(input_layer, filter, block_name, position='EN'):
    # block_num = str(block_name)
    # print("block_name",block_name)
    dilated_1 = Conv2D(filter, 3, dilation_rate=1, padding='same',
                       name='DilatedBlock_rate1_{}_{}'.format(block_name, position))(input_layer)
    dilated_2 = Conv2D(filter, 3, dilation_rate=2, padding='same',
                       name='DilatedBlock_rate2_{}_{}'.format(block_name, position))(input_layer)
    dilated_3 = Conv2D(filter, 3, dilation_rate=3, padding='same',
                       name='DilatedBlock_rate3_{}_{}'.format(block_name, position))(input_layer)
    dilated_4 = Conv2D(filter, 3, dilation_rate=4, padding='same',
                       name='DilatedBlock_rate4_{}_{}'.format(block_name, position))(input_layer)
    dilated_5 = Conv2D(filter, 3, dilation_rate=5, padding='same',
                       name='DilatedBlock_rate5_{}_{}'.format(block_name, position))(input_layer)

    concat = Concatenate(name='DilatedBlock_concat_{}_{}'.format(block_name, position))(
        [dilated_1, dilated_2, dilated_3, dilated_4, dilated_5])
    # print('concat', concat.shape)
    output = Conv2D(filter, 3, padding='same', name="DilatedBlock_Conv2D_{}_{}".format(block_name, position))(concat)
    # print('output', output.shape)

    return output


''''
Encoder / Decoder
'''


def DecodeLayer(input_layer, filter=16, block_num=5, blocks=6, non_linear=True, var=0.03,
                first_block='Plain', second_block='Plain', activation_name = 'relu', position='NB', droppath_rate=0):
    """
    Conv - ResBlockScale - Conv2DTranspose
    """
    str_block_num = str(block_num)

    intput = BasicConv(input_layer, filter=filter, block_name=block_num, position=position)
    for i in range(blocks):
        dScale = getScale(blocks=blocks, i=(i + blocks + 1), non_linear=non_linear, var=var, encode=False)
        if i == 0:
            layers = ResBlockScale(intput, filter=filter, block_name=f"{block_num}_{i}", scale=dScale, activation_name = activation_name,
                                   block_type=first_block, position=position, droppath_rate=droppath_rate)
        elif i % 2 == 1:
            layers = ResBlockScale(layers, filter=filter, block_name=f"{block_num}_{i}", scale=dScale, activation_name = activation_name,
                                   block_type=second_block,  position=position, droppath_rate=droppath_rate)
        else:
            layers = ResBlockScale(layers, filter=filter, block_name=f"{block_num}_{i}", scale=dScale, activation_name = activation_name,
                                   block_type=first_block,  position=position, droppath_rate=droppath_rate)

    output = BasicConv(layers, filter=filter, transpose=True, block_name=f"Trans_{block_num}", position=position)

    return output


def EncodeLayer(input_layer, downsample_input=None, filter=16, block_num=5, blocks=6, non_linear=True, var=0.03, dilated = True,
                first_block='Plain', second_block='Plain', position='EN', activation_name = 'relu', droppath_rate=0):
    block_num = str(block_num)

    if dilated:
        dil_conv = DilatedBlock(input_layer, filter=filter, block_name=block_num)
    else: 
        dil_conv = BasicConv(input_layer, filter=filter, block_name=block_num)

    if block_num != "1":

        SCM_layer = SCM(downsample_input, filter=filter, block_num=block_num)
        FAM_layer = FAM(SCM_layer, dil_conv, filter=filter, block_num=block_num)

        res_input = FAM_layer
    else:
        res_input = dil_conv

    for i in range(blocks):
        eScale = getScale(blocks=blocks, i=i, non_linear=non_linear, var=var)
        if i == 0:
            layers = ResBlockScale(res_input, filter=filter, block_name=f"{block_num}_{i}", scale=eScale,
                                   block_type=first_block, position=position, activation_name = activation_name, droppath_rate=droppath_rate)
        elif i % 2 == 1:
            layers = ResBlockScale(layers, filter=filter, block_name=f"{block_num}_{i}", scale=eScale,
                                   block_type=second_block, position=position, activation_name = activation_name, droppath_rate=droppath_rate)
        else:
            layers = ResBlockScale(layers, filter=filter, block_name=f"{block_num}_{i}", scale=eScale,
                                   block_type=first_block, position=position, activation_name = activation_name, droppath_rate=droppath_rate)

    return layers


def BottleNeck(input_layer, downsample_input=None, filter=16, block_num=5, blocks=6, non_linear=True, var=0.03, dilated = True,
               first_block='Plain', second_block='Plain', position='BottleNeck', atten =None, activation_name = 'relu', droppath_rate=0):
    block_num = str(block_num)

    if dilated:
        bottleneck = DilatedBlock(input_layer, filter=filter, block_name=4)
    else:
        bottleneck = BasicConv(input_layer, filter=filter, block_name=block_num)
    SCM_layer = SCM(downsample_input, filter=filter, block_num=block_num)

    if atten=='SEBlock':

        concat = Concatenate()([SCM_layer, bottleneck])
        concat =Conv2D(filter, (3, 3), padding='same', name='bottleneck_concat')(concat)

        FAM_layer = se_block(concat, 8)

    else:
        FAM_layer = FAM(SCM_layer, bottleneck, filter=filter)

    for i in range(2 * blocks):
        if i > blocks - 1:
            neck_scale = getScale(blocks=blocks, i=(i + 1), non_linear=non_linear, var=var, encode=False)
        else:
            neck_scale = getScale(blocks=blocks, i=i, non_linear=non_linear, var=var)

        if i == 0:
            bottleneck = ResBlockScale(FAM_layer, filter=filter, block_name=f"{i}", scale=neck_scale,
                                       block_type=first_block, position=position, activation_name = activation_name, droppath_rate=droppath_rate)
            print('i', i, f'Plain\t', neck_scale)
        elif i % 2 == 1:
            bottleneck = ResBlockScale(bottleneck, filter=filter, block_name=f"{i}", scale=neck_scale,
                                       block_type=second_block, position=position, activation_name = activation_name, droppath_rate=droppath_rate)
            print('i', i, f'Plain\t', neck_scale)

        else:
            bottleneck = ResBlockScale(bottleneck, filter=filter, block_name=f"{i}", scale=neck_scale,
                                       block_type=first_block, position=position, activation_name = activation_name, droppath_rate=droppath_rate)
            print('i', i, f'Plain\t', neck_scale)

    return bottleneck

