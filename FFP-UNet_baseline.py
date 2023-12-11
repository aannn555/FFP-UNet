import numpy as np
import os
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf

from tensorflow.keras import backend as K

from blurpool import BlurPool2D
from BasicBlock import SCM, FAM

# from MimoUNet import * 

def BasicConv(input_layer, filter = 16, dropout=True, kernel_size=(3,3), batch_norm=False, block_num=1, transpose=False, act_func = 'relu'):
    '''
    conv - BN - Activation - Dropout
    '''

    if transpose == True:
        conv = Conv2DTranspose(filter, kernel_size=kernel_size, padding="same", name="BasicTransConv_{}".format(block_num))(input_layer)
    else:
        conv = Conv2D(filter, kernel_size=kernel_size, padding="same", name="BasicConv_{}".format(block_num))(input_layer)
    
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)

    conv = Activation(act_func)(conv)

    if dropout is True:
        conv = Dropout(0.2)(conv)

    return conv

def ResBlockScale(input_layer, filter = 16, kernel_size=(3,3), batch_norm=False, block_num=1, scale=3, act_func = 'relu'):
    '''
    conv - BN - Activation - conv - BN - Activation 
                                        - shortcut  - BN - shortcut
    '''
    str_block_num = str(block_num)

    conv = Conv2D(filter, kernel_size=kernel_size, padding="same",name="ResBlock{}_a".format(str_block_num))(input_layer)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation(act_func)(conv)
    # print('conv2',conv.shape)
    

    conv = Conv2D(filter, kernel_size=kernel_size, padding="same",name="ResBlock{}_b".format(str_block_num))(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation(act_func)(conv)

    shortcut = tf.keras.layers.Lambda(lambda inputs, scale: inputs[0]+ inputs[1] * scale,
               output_shape = K.int_shape(conv)[1:],
               arguments = {'scale': scale}, name="Shortcut_{}_{}".format(str_block_num, scale))([input_layer,conv])

    res_path = Activation(act_func)(shortcut)    #Activation after addition with shortcut (Original residual block)
    return res_path

def SCM(input_layer, filter=16, block_num = 1, activation_name = 'relu'):
    str_block_num = str(block_num)

    conv = Conv2D(filter, (3,3), strides=(1,1), padding="same", name = 'SCM_{}'.format(str_block_num))(input_layer) # 3x3
    conv = Activation(activation_name)(conv)
    conv = Conv2D(filter, (1,1), strides=(1,1), padding="same")(conv) # 1x1
    conv = Activation(activation_name)(conv)
    conv = Conv2D(filter, (3,3), strides=(1,1), padding="same")(conv) # 3x3
    conv = Activation(activation_name)(conv)
    conv = Conv2D(filter, (1,1), strides=(1,1), padding="same")(conv) # 1x1
    conv = Activation(activation_name)(conv)

    concat = Concatenate()([input_layer, conv])
    conv_last = Conv2D(filter, (1,1), strides=(1,1), padding="same", name = 'SCM_last_{}'.format(str_block_num))(concat) # 1x1

    return conv_last


def FAM(input, pre_conv, filter=32, name="FAM"):
    '''
    input : SCMout
    pre_conv : EB out (follow convolution)
    '''
    multiplied = Multiply()([input, pre_conv])
    multiplied = Conv2D(filter, (3,3), padding='same', name="FAM_{}".format(filter))(multiplied)
    output = Add()([multiplied, pre_conv])  

    output = Conv2D(filter, (1, 1), padding='same')(output) # debug
    output = BatchNormalization()(output) # debug

    # output = Activation("relu")(output)

    return output

def getGaussianScale(var, blocks, i):
    scale = 1 - np.exp(-var * np.power(blocks - i, 2))
    return scale

def DilatedBlock(input_layer, filter, block_num, name = 'DilatedBlock'):
    block_num = str(block_num)

    dilated_1 = Conv2D(filter, 3, dilation_rate=1, padding='same', name='{}{}_rate1'.format(name, block_num))(input_layer)
    dilated_2 = Conv2D(filter, 3, dilation_rate=2, padding='same', name='{}{}_rate2'.format(name, block_num))(input_layer)
    dilated_3 = Conv2D(filter, 3, dilation_rate=3, padding='same', name='{}{}_rate3'.format(name, block_num))(input_layer)
    dilated_4 = Conv2D(filter, 3, dilation_rate=4, padding='same', name='{}{}_rate4'.format(name, block_num))(input_layer)
    dilated_5 = Conv2D(filter, 3, dilation_rate=5, padding='same', name='{}{}_rate5'.format(name, block_num))(input_layer)
    
    concat = Concatenate()([dilated_1, dilated_2, dilated_3, dilated_4, dilated_5])
    print('concat', concat.shape)
    output = Conv2D(filter, 3, padding='same')(concat)
    print('output', output.shape)

    return output


def UpSamplingModule(input_layer, filter, block_num, up_rate = (2, 2), interpolation="bilinear", act_func = 'relu', name = 'UpSamplingModule'):
    block_num = str(block_num)
    conv = Conv2D(filter, 1, padding="same", name="UpSamplingModule{}_start".format(block_num))(input_layer)
    conv = Activation(act_func)(conv)
    conv = Conv2D(filter, 3, padding="same")(conv)
    conv = Activation(act_func)(conv)
    conv = UpSampling2D(up_rate, interpolation= interpolation)(conv)
    conv = Conv2D(filter, 1, padding="same")(conv)
    # print('mainPath', conv.shape)

    shortcut = UpSampling2D(up_rate, interpolation= interpolation)(input_layer)
    shortcut = Conv2D(filter, 1, padding="same")(shortcut)
    # print('shortcut', shortcut.shape)

    output = add([shortcut, conv])

    return output

def DownSamplingModule(input_layer, filter, block_num, act_func = 'relu',name = 'DownSamplingModule'):
    # with Antialiasing DownSampling

    block_num = str(block_num)

    conv = Conv2D(filter, 1, padding="same", name="UpSamplingModule{}_start".format(block_num))(input_layer)
    conv = Activation(act_func)(conv)
    conv = Conv2D(filter, 3, padding="same")(conv)
    conv = Activation(act_func)(conv)

    conv = BlurPool2D()(conv)
    conv = Conv2D(filter, 1, padding="same")(conv)
    print('mainPath', conv.shape)

    shortcut = BlurPool2D()(input_layer)
    shortcut = Conv2D(filter, 1, padding="same")(shortcut)
    print('shortcut', shortcut.shape)

    output = add([shortcut, conv])

    return output


# Best !!!
def MimoUNet_v5_sample_v1(input_size=None, name='MimoUNet_v5_sample_v1', input_channel=2, blocks = 1, 
                          alpha = 0.03, interpolation = 'bilinear', act_func = 'relu'):

    """
    Include:
        [@ Middle(U-Shape):
            - UpSample Module (Bilinear)
            - DownSample Module (MaxBlurPool = Antialiasing)]
        - Parallel Dilated
        - Gaussian Scale = 0.03
    """ 

    filters = [16, 32, 64, 128]

    input1 = layers.Input(shape=(None, None, input_channel), name="input1")
    
    conv1 = DilatedBlock(input1, filter = filters[0], block_num = 1)
    for i in range(blocks):
        eScale = getGaussianScale(alpha, blocks, i)
        if i == 0:
            conv1_layers = ResBlockScale(conv1, filter=filters[0], block_num=f"EB1-{i}", scale= eScale, act_func = act_func)
        else:
            conv1_layers = ResBlockScale(conv1_layers, filter=filters[0], block_num=f"EB1-{i}", scale=eScale, act_func = act_func)
    # pool1 = BlurPool2D()(conv1_layers)
    pool1 = DownSamplingModule(conv1_layers, filters[0], block_num = 'pool1', act_func = act_func)
    
    B2 =  Conv2D(filters[1], (3,3), strides=2, padding='same')(input1)
    conv2 = DilatedBlock(pool1, filter=filters[1], block_num=2)
    SCM2 = SCM(B2, filter=filters[1], block_num =2, activation_name = act_func)
    FAM2 = FAM(SCM2, conv2, filter=filters[1])
    for i in range(blocks):
        eScale = getGaussianScale(alpha, blocks, i)
        if i == 0:
            conv2_layers = ResBlockScale(FAM2, filter=filters[1], block_num=f"EB2-{i}", scale= eScale, act_func = act_func)
        else:
            conv2_layers = ResBlockScale(conv2_layers, filter=filters[1], block_num=f"EB2-{i}", scale=eScale, act_func = act_func)
    # pool2 = BlurPool2D()(conv2_layers)
    pool2 = DownSamplingModule(conv2_layers, filters[1], block_num = 'pool2', act_func = act_func)


    # B3 = layers.experimental.preprocessing.Resizing(shape_input1[1]//4, shape_input1[2]//4)(input1)
    B3 =  Conv2D(filters[2], (3,3), strides=2, padding='same')(B2)
    conv3 = DilatedBlock(pool2, filter=filters[2], block_num=3)
    SCM3 = SCM(B3, filter=filters[2], block_num=3, activation_name = act_func)
    FAM3 = FAM(SCM3, conv3, filter=filters[2])
    for i in range(blocks):
        eScale = getGaussianScale(alpha, blocks, i)
        if i == 0:
            conv3_layers = ResBlockScale(FAM3, filter=filters[2], block_num=f"EB3-{i}", scale= eScale, act_func = act_func)
        else:
            conv3_layers = ResBlockScale(conv3_layers, filter=filters[2], block_num=f"EB3-{i}", scale=eScale, act_func = act_func)
    # pool3 = BlurPool2D()(conv3_layers)
    pool3 = DownSamplingModule(conv3_layers, filters[2], block_num = 'pool3', act_func = act_func)


    # Bottleneck
    print('******* Bottleneck *******')
    # B4 = layers.experimental.preprocessing.Resizing(shape_input1[1]//8, shape_input1[2]//8)(input1)
    B4 =  Conv2D(filters[3], (3,3), strides=2, padding='same')(B3)
    bottleneck = DilatedBlock(pool3, filter=filters[3], block_num=4)

    SCM4 = SCM(B4, filter=filters[3], block_num=4, activation_name = act_func)
    FAM4 = FAM(SCM4, bottleneck, filter=filters[3])

    for i in range(2*blocks):
        neck_scale = getGaussianScale(alpha, blocks, i)
        if i ==0:
            bottleneck = ResBlockScale(FAM4, filter=filters[3], block_num=401, scale =neck_scale, act_func = act_func)
            print('i',i,'neck_scale',neck_scale)
        elif i > blocks-1:
            neck_scale = -getGaussianScale(alpha, blocks, (i+1))
            bottleneck = ResBlockScale(bottleneck, filter=filters[3], block_num=f"bottleneck-{i}", scale =neck_scale, act_func = act_func)
            print('i',i,'neck_scale',neck_scale)
        else:
            bottleneck = ResBlockScale(bottleneck, filter=filters[3], block_num=f"bottleneck-{i}", scale =neck_scale, act_func = act_func)
            print('i',i,'neck_scale',neck_scale)

    bottleneck = BasicConv(bottleneck, filter=filters[3], block_num=403, transpose=True, act_func = act_func)
    print('bottleneck',bottleneck.shape)

    
    print('******* Decode *******')

    db3 = UpSamplingModule(bottleneck, filters[3], block_num = 'db3', interpolation=interpolation, act_func = act_func)
    print('db3', db3.shape)
    db3 = Concatenate()([db3, conv3_layers])

    db3 = BasicConv(db3, filter=filters[2], block_num=5, act_func = act_func)
    for i in range(blocks):
        dScale = -getGaussianScale(alpha, blocks, (i+blocks+1))
        if i == 0:
            db3_layers = ResBlockScale(db3, filter=filters[2], block_num=f"DB3-{i}", scale= dScale, act_func = act_func)
        else:
            db3_layers = ResBlockScale(db3_layers, filter=filters[2], block_num=f"DB3-{i}", scale=dScale, act_func = act_func)
    db3 = BasicConv(db3_layers, filter=filters[2], transpose=True, block_num="Trans5", act_func = act_func)

    # db2 = UpSampling2D((2, 2), interpolation="bilinear")(db3)
    db2 = UpSamplingModule(db3, filters[2], block_num = 'db2', interpolation =interpolation, act_func = act_func)
    db2 = Concatenate()([db2, conv2_layers])

    db2 = BasicConv(db2, filter=filters[1], block_num=6, act_func = act_func)
    for i in range(blocks):
        dScale = -getGaussianScale(alpha, blocks, (i+blocks+1))
        if i == 0:
            db2_layers = ResBlockScale(db2, filter=filters[1], block_num=f"DB2-{i}", scale= dScale, act_func = act_func)
        else:
            db2_layers = ResBlockScale(db2_layers, filter=filters[1], block_num=f"DB2-{i}", scale=dScale, act_func = act_func)
    db2 = BasicConv(db2_layers, filter=filters[1], block_num="Trans6", transpose=True, act_func = act_func)

    # db1 = UpSampling2D((2, 2), interpolation="bilinear")(db2)
    db1 = UpSamplingModule(db2, filters[1], block_num = 'db1', interpolation =interpolation, act_func = act_func)

    db1 = Concatenate()([db1, conv1_layers])

    db1 = BasicConv(db1, filter=filters[0], block_num=7, act_func = act_func)
    for i in range(blocks):
        dScale = -getGaussianScale(alpha, blocks, (i+blocks+1))
        if i == 0:
            db1_layers = ResBlockScale(db1, filter=filters[0], block_num=f"DB1-{i}", scale= dScale, act_func = act_func)
        else:
            db1_layers = ResBlockScale(db1_layers, filter=filters[0], block_num=f"DB1-{i}", scale=dScale, act_func = act_func)

    db1 = BasicConv(db1_layers, filter=filters[0], block_num="Trans7", act_func = act_func)

    conv81 = UpSampling2D(size=(8, 8), interpolation = interpolation)(bottleneck)
    
    conv82 = UpSampling2D(size=(4, 4), interpolation = interpolation)(db3)
    conv83 = UpSampling2D(size=(2, 2), interpolation = interpolation)(db2)
    conv8 = concatenate([conv81, conv82, conv83, db1], axis=-1)
    conv8 = Conv2D(input_channel, (1, 1))(conv8)

    output=Add()([input1,conv8])
    output = Conv2D(input_channel, (1, 1))(output)
    output = BatchNormalization(axis=3)(output)
    output = Activation('sigmoid')(output)
    
    model = Model(input1, output)

    return model



if __name__ == '__main__':
    model = MimoUNet_v5_sample_v1(blocks=5 ,input_channel=1, act_func = 'relu')
    # model.summary()

    inputs = tf.random.uniform((32,40,40,1), maxval=1)
    print(inputs.shape)
    output = model.predict(inputs)
    samples_to_predict = np.array(output)
    print(samples_to_predict.shape)

    os.makedirs('./plotModel/', exist_ok=True)

    dot_img_file = './plotModel/'+ MimoUNet_v5_sample_v1.__name__.split('.')[0] + '.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    print(MimoUNet_v5_sample_v1.__name__)
    