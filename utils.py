from re import T
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np

from blurpool import BlurPool2D

from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid


'''
  Droppath
'''

class StochasticDepth(layers.Layer):
    """Stochastic Depth module.

    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.

    References:
      - https://github.com/rwightman/pytorch-image-models

    Args:
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].

    Returns:
      Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config



'''
   Scale Mechanism
'''


def getGaussianScale(var, blocks, i):
    scale = 1 - np.exp(-var * np.power(blocks - i, 2))
    return scale


def getLinearDecayScale(alpha=0.01, blocks=5, i=0):
    total_scale = blocks * alpha
    scale = total_scale - (i * alpha)

    return scale


def getScale(blocks=5, i=0, non_linear=True, encode=True, var=0.03, alpha=0.01):
    if non_linear:
        if encode:
            scale = getGaussianScale(var, blocks, i)
        else:
            scale = -getGaussianScale(var, blocks, i)
    else:
        scale = getLinearDecayScale(alpha, blocks, i)
    return scale


'''
    Sampling Module
'''


def UpSamplingModule(input_layer, filter, block_num, up_rate=2, name='UpSamplingModule', position='NB', activation_name = 'relu',
                     interpolation='bilinear'):
    block_num = str(block_num)

    conv = Conv2D(filter, 1, padding="same", name="{}_{}_1_{}".format(name, block_num, position))(input_layer)
    # conv = ReLU(name="ReLU_1_{}_{}_{}".format(name, block_num, position))(conv)
    conv = Activation(activation_name)(conv)

    conv = Conv2D(filter, 3, padding="same", name="{}_{}_2_{}".format(name, block_num, position))(conv)
    # conv = ReLU(name="ReLU_2_{}_{}_{}".format(name, block_num, position))(conv)
    conv = Activation(activation_name)(conv)


    if interpolation == 'bilinear':
        conv = UpSampling2D(up_rate, interpolation=interpolation,
                            name="UpSampling2D_1_{}_{}_{}".format(name, block_num, position))(conv)
    else:
        conv = tf.image.resize(conv, [tf.shape(conv)[1] * up_rate, tf.shape(conv)[2] * up_rate], method=interpolation,
                               name="UpSampling2D_1_{}_{}_{}".format(name, block_num, position))

    conv = Conv2D(filter, 1, padding="same", name="{}_{}_last_{}".format(name, block_num, position))(conv)

    if interpolation == 'bilinear':
        shortcut = UpSampling2D(up_rate, interpolation=interpolation,
                                name="UpSampling2D_shortcut_{}_{}_{}".format(name, block_num, position))(input_layer)
    else:
        shortcut = tf.image.resize(input_layer,
                                   [tf.shape(input_layer)[1] * up_rate, tf.shape(input_layer)[2] * up_rate],
                                   method=interpolation,
                                   name="UpSampling2D_shortcut_{}_{}_{}".format(name, block_num, position))

    shortcut = Conv2D(filter, 1, padding="same",
                      name="Conv2D_shortcut_{}_{}_{}".format(name, block_num, position))(shortcut)

    output = add([shortcut, conv])

    return output


def DownSamplingModule(input_layer, filter, block_num, name='DownSamplingModule', activation_name = 'relu', position='EN'):
    # with Antialiasing DownSampling

    block_num = str(block_num)

    conv = Conv2D(filter, 1, padding="same", name="{}_{}_1_{}".format(name, block_num, position))(input_layer)
    # conv = ReLU(name="ReLU_1_{}_{}_{}".format(name, block_num, position))(conv)
    conv = Activation(activation_name)(conv)

    conv = Conv2D(filter, 3, padding="same", name="{}_{}_2_{}".format(name, block_num, position))(conv)
    # conv = ReLU(name="ReLU_2_{}_{}_{}".format(name, block_num, position))(conv)
    conv = Activation(activation_name)(conv)


    conv = BlurPool2D(name="BlurPool2D_1_{}_{}_{}".format(name, block_num, position))(conv)
    conv = Conv2D(filter, 1, padding="same", name="{}_{}_last_{}".format(name, block_num, position))(conv)
    # print('mainPath', conv.shape)

    shortcut = BlurPool2D(name="BlurPool2D_shortcut_{}_{}_{}".format(name, block_num, position))(input_layer)
    shortcut = Conv2D(filter, 1, padding="same", name="{}_{}_shortcut_{}".format(name, block_num, position))(shortcut)
    # print('shortcut', shortcut.shape)

    output = add([shortcut, conv], name="Add_{}_{}_{}".format(name, block_num, position))

    return output


'''
    Attention Module
'''


class SKNet(Model):
    def __init__(self):
        super(SKNet, self).__init__()

        # Block 1
        self.conv1_a = Conv2D(filters=16, kernel_size=3, padding="SAME")
        self.bn1_a = BatchNormalization()
        self.conv1_b = Conv2D(filters=16, kernel_size=3, padding="SAME")
        self.bn1_b = BatchNormalization()

        self.fc1 = Dense(8, activation=None)
        self.bn1_fc = BatchNormalization()

        self.fc1_a = Dense(16, activation=None)
      
        self.maxpool = GlobalAveragePooling2D()

    def call(self, x, y, verbose=False, relu=True, debug= False, concatInput =False):

        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = x.shape[channel_axis]

        if verbose:
            print('SKNet')
            print(x.shape)
            print('channel', channel)

        """ Conv-1 """
        # Split-1
        if relu:
            u1_a = tf.keras.activations.relu(self.bn1_a(self.conv1_a(x)))
            u1_b = tf.keras.activations.relu(self.bn1_b(self.conv1_b(y)))
        else:
            u1_a = tf.keras.layers.LeakyReLU()(self.bn1_a(self.conv1_a(x)))
            u1_b = tf.keras.layers.LeakyReLU()(self.bn1_b(self.conv1_b(y)))
        
        # Fuse-1
        u1 = u1_a + u1_b
        print('u1',u1.shape) # (None, 40, 40, 16)
        s1 = tf.math.reduce_sum(u1, axis=(1, 2))

        if concatInput:
            print('u1_a', u1_a.shape)
            print('u1_b', u1_b.shape)
            s1 = Concatenate()([u1_a , u1_b])
            s1 = tf.math.reduce_sum(s1, axis=(1, 2))
            print('s1', s1.shape)
            print('Concat Input!!!')

        if relu:
            z1 = tf.keras.activations.relu(self.bn1_fc(self.fc1(s1)))
        else:
            z1 = tf.keras.layers.LeakyReLU()(self.bn1_fc(self.fc1(s1)))
            print('LeakyReLU!')


        print('z1', z1.shape)

        # Select-1
        a1 = tf.keras.activations.softmax(self.fc1_a(z1))
        a1 = tf.expand_dims(a1, 1)
        a1 = tf.expand_dims(a1, 1)
        b1 = 1 - a1
        if debug:
            print('debug')
            a_feature = u1_a * a1
            b_feature = u1_b * b1
            print('a_feature', a_feature.shape)
            print('b_feature', b_feature.shape)
            out = Concatenate()([a_feature, b_feature])
            print('out', out.shape)
        else:
            # a_feature = u1_a * a1
            # b_feature = u1_b * b1
            # print('a_feature', a_feature.shape)
            # print('b_feature', b_feature.shape)
            out = (u1_a * a1) + (u1_b * b1)
        if verbose: print('[v1.shape]', out.shape)


        return out
    
class SKNet_v2(Model):
    def __init__(self):
        super(SKNet_v2, self).__init__()

        # Block 1
        self.conv1_a = Conv2D(filters=16, kernel_size=3, padding="SAME")
        self.bn1_a = BatchNormalization()
        self.conv1_b = Conv2D(filters=16, kernel_size=3, padding="SAME")
        self.bn1_b = BatchNormalization()
        self.conv1_b_2 = Conv2D(filters=16, kernel_size=5, padding="SAME")
        self.bn1_b_2 = BatchNormalization()


        self.fc1 = Dense(8, activation=None)
        self.bn1_fc = BatchNormalization()

        self.fc1_a = Dense(16, activation=None)
      
        self.maxpool = GlobalAveragePooling2D()

    def call(self, x, y, verbose=False, relu=False, debug= False):
        
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = x.shape[channel_axis]

        if verbose:
            
            print(x.shape)
            print('channel', channel)

        """ Conv-1 """
        # Split-1
        if relu:
            u1_a = tf.keras.activations.relu(self.bn1_a(self.conv1_a(x)))
            u1_b = tf.keras.activations.relu(self.bn1_b(self.conv1_b(y)))
        else:
            u1_a = tf.keras.layers.LeakyReLU()(self.bn1_a(self.conv1_a(x)))
            u1_b = tf.keras.layers.LeakyReLU()(self.bn1_b(self.conv1_b(y)))
            u1_b_2 = tf.keras.layers.LeakyReLU()(self.bn1_b_2(self.conv1_b_2(y)))

        # Fuse-1
        u1 = u1_a + u1_b + u1_b_2
        s1 = tf.math.reduce_sum(u1, axis=(1, 2))
        if relu:
            z1 = tf.keras.activations.relu(self.bn1_fc(self.fc1(s1)))
        else:
            z1 = tf.keras.layers.LeakyReLU()(self.bn1_fc(self.fc1(s1)))
            print('LeakyReLU!')


        # Select-1
        a1 = tf.keras.activations.softmax(self.fc1_a(z1))
        a1 = tf.expand_dims(a1, 1)
        a1 = tf.expand_dims(a1, 1)
        b1 = 1 - a1
        if debug: # concat
            print('debug')
            a_feature = u1_b * a1
            b_feature = u1_b_2 * b1
            print('a_feature', a_feature.shape)
            print('b_feature', b_feature.shape)
            out = Concatenate()([a_feature, b_feature])
            print('out', out.shape)
        else:
            out = (u1_b * a1) + (u1_b_2 * b1)
        if verbose: print('[v1.shape]', out.shape)


        return out



def se_block(input_feature, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel)
    se_feature = Dense(channel // ratio,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel // ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel)
    print('se_feature', se_feature.shape)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature


def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='sigmoid',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])



