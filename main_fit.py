# -*- coding: utf-8 -*-
from doctest import debug
import os
from re import T
import cv2
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras

# from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
import sys
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

from glob import glob
import matplotlib
import matplotlib.pyplot as plt

from loss_set import *
from LossLearningRateScheduler import *

# from SIFTAttNet import *
# from MimoUNet_v5_sample import *
from FPNResUNet import *


import argparse

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

epochs = 80
batch_size = 32
initial_learning_rate = 1e-3 # 1e-3
save_name = '0822_SIFTAttNet_v4_encoder_CBAM'
notGaussian = False
save_path = './checkpoint/' + save_name

lr_csv_path = save_path + '.csv'
os.makedirs(save_path, exist_ok=True)


'''
1. Load Data
X: noise
Y: without noise
'''

path_x_train = sorted(glob(
    '/data/FocalTech/nasic9395_0606_aug_v3_testtool_siftMap/train/non_binary/x_train/*.bmp'))
path_x_train_binary = sorted(glob(
    "/data/FocalTech/nasic9395_0606_aug_v3_testtool_siftMap/train/binary/x_train/*.bmp"))
path_x_map = sorted(glob(
    "/data/FocalTech/nasic9395_0606_aug_v3_testtool_siftMap/train/siftMap/*.bmp"))

path_y_train = sorted(glob(
    '/data/FocalTech/nasic9395_0606_aug_v3_testtool_siftMap/train/non_binary/y_train/*.bmp'))
path_y_train_binary = sorted(glob(
    "/data/FocalTech/nasic9395_0606_aug_v3_testtool_siftMap/train/binary/y_train/*.bmp"))

path_x_val = sorted(glob(
    '/data/FocalTech/nasic9395_0606_aug_v3_testtool_siftMap/val/non_binary/x_val/*.bmp'))
path_x_val_binary = sorted(glob(
    "/data/FocalTech/nasic9395_0606_aug_v3_testtool_siftMap/val/binary/x_val/*.bmp"))
path_y_map = sorted(glob(
    "/data/FocalTech/nasic9395_0606_aug_v3_testtool_siftMap/val/siftMap/*.bmp"))

path_y_val = sorted(glob(
    '/data/FocalTech/nasic9395_0606_aug_v3_testtool_siftMap/val/non_binary/y_val/*.bmp'))
path_y_val_binary = sorted(glob(
    "/data/FocalTech/nasic9395_0606_aug_v3_testtool_siftMap/val/binary/y_val/*.bmp"))

print("LOAD DATA!")
print(len(path_x_train))
print(len(path_y_train))
print(len(path_x_map))

print(len(path_x_val))
print(len(path_y_val))
print(len(path_y_map))

assert len(path_x_train) == len(path_x_train_binary), 'Training amount is not equal'
assert len(path_y_train) == len(path_y_train_binary), 'Training amount is not equal'
assert len(path_y_train) == len(path_x_map), 'train map amount is not equal'

assert len(path_x_val) == len(path_x_val_binary), 'Val amount is not equal'
assert len(path_y_val) == len(path_y_val_binary), 'Val amount is not equal'
assert len(path_y_val) == len(path_y_map), 'train map amount is not equal'

print("LOAD DATA!")
print(len(path_x_train))
print(len(path_y_train))
print(len(path_x_val))
print(len(path_y_val))


def load_path(path_name=path_x_train):
    filenames = []
    file_num = len(path_name)
    for img in path_name:
        filenames.append(img)
    return filenames


x_train = load_path(path_x_train)
x_train_binary = load_path(path_x_train_binary)

y_train = load_path(path_y_train)
y_train_binary = load_path(path_y_train_binary)

x_val = load_path(path_x_val)
x_val_binary = load_path(path_x_val_binary)

y_val = load_path(path_y_val)
y_val_binary = load_path(path_y_val_binary)

x_map = load_path(path_x_map)
y_map = load_path(path_y_map)


'''
2. Data Preprocess
- Resize 
'''
def resize_image(filenames):
    #####################################
    ## Mirror Symmetry Padding ##
    #####################################
    array_of_img = []
    for filename in filenames:
        img = cv2.imread(filename, 0)
        padding_img = np.full((40, 40), 0)  # original size: 100 x 36

        dif_h = padding_img.shape[0] - img.shape[0]
        dif_w = padding_img.shape[1] - img.shape[1]

        pad_h = dif_h // 2
        pad_w = dif_w // 2

        # Space4,6
        Space4 = np.flip(img, axis=1)

        # append Space4,5,6
        centerRow = np.append(Space4, img, axis=1)
        centerRow = np.append(centerRow, Space4, axis=1)

        # create up&down row
        UpDownRow = np.flip(centerRow, axis=0)

        # append Space1,2,3
        upRow = np.append(UpDownRow, centerRow, axis=0)

        # append Space7,8,9
        fullPadding = np.append(upRow, UpDownRow, axis=0)

        if (dif_h % 2 == 1) & (dif_w % 2 != 1):
            cropArea = fullPadding[img.shape[0] - pad_h:2 * img.shape[0] + (pad_h + 1),
                       img.shape[1] - pad_w:2 * img.shape[1] + pad_w]
        elif (dif_h % 2 != 1) & (dif_w % 2 == 1):
            cropArea = fullPadding[img.shape[0] - pad_h:2 * img.shape[0] + pad_h,
                       img.shape[1] - pad_w:2 * img.shape[1] + (pad_w + 1)]
        elif (dif_h % 2 == 1) & (dif_w % 2 == 1):
            cropArea = fullPadding[img.shape[0] - pad_h:2 * img.shape[0] + (pad_h + 1),
                       img.shape[1] - pad_w:2 * img.shape[1] + (pad_w + 1)]
        elif (dif_h % 2 != 1) & (dif_w % 2 != 1):
            cropArea = fullPadding[img.shape[0] - pad_h:2 * img.shape[0] + pad_h,
                       img.shape[1] - pad_w: 2 * img.shape[1] + pad_w]
        # print(cropArea.shape)
        if cropArea.shape != (40, 40):
            print("fail")
            print(filename)
            print(img.shape)
            print("*" * 50)
        else:
            array_of_img.append(cropArea)

    Image = np.array(array_of_img, dtype=np.float32)

    return Image / 255.0


x_train_padding = resize_image(x_train)
x_train_binary_padding = resize_image(x_train_binary)

y_train_padding = resize_image(y_train)
y_train_binary_padding = resize_image(y_train_binary)

x_val_padding = resize_image(x_val)
x_val_binary_padding = resize_image(x_val_binary)

y_val_padding = resize_image(y_val)
y_val_binary_padding = resize_image(y_val_binary)

x_map_padding = resize_image(x_map)
y_map_padding = resize_image(y_map)

dataset_xtrain_single = np.expand_dims(x_train_padding, axis=-1)
dataset_xtrain_binary_single = np.expand_dims(x_train_binary_padding, axis=-1)

dataset_ytrain_single = np.expand_dims(y_train_padding, axis=-1)
dataset_ytrain_binary_single = np.expand_dims(y_train_binary_padding, axis=-1)

dataset_xval_single = np.expand_dims(x_val_padding, axis=-1)
dataset_xval_binary_single = np.expand_dims(x_val_binary_padding, axis=-1)

dataset_yval_single = np.expand_dims(y_val_padding, axis=-1)
dataset_yval_binary_single = np.expand_dims(y_val_binary_padding, axis=-1)

dataset_x_map = np.expand_dims(x_map_padding, axis=-1)
dataset_y_map = np.expand_dims(y_map_padding, axis=-1)


'''
2. Data Preprocess
- (Optional)notGaussian: 把map變成只有 0,1
- (Optional)combine_channels: 如果要跟binary一起訓練會用到

'''
def convertMap(map):
    map[map > 0] = 1
    return map

if notGaussian == True:
    print('notGaussian '*5)
    x_map_padding = convertMap(x_map_padding)
    y_map_padding = convertMap(y_map_padding)


def combine_channels(path_name=path_x_train, binary_path_name=x_train_binary):
    num = len(path_name)
    needed_multi_channel_img = np.zeros((num, 40, 40, 2))

    for i in range(num):
        needed_multi_channel_img[i, :, :, 0] = path_name[i]
        needed_multi_channel_img[i, :, :, 1] = binary_path_name[i]
    return needed_multi_channel_img


dataset_xtrain = combine_channels(x_train_padding, x_train_binary_padding)
print('dataset_xtrain', dataset_xtrain.shape)
dataset_ytrain = combine_channels(y_train_padding, y_train_binary_padding)
print('dataset_ytrain', dataset_ytrain.shape)
dataset_xval = combine_channels(x_val_padding, x_val_binary_padding)
print('dataset_xval', dataset_xval.shape)
dataset_yval = combine_channels(y_val_padding, y_val_binary_padding)
print('dataset_yval', dataset_yval.shape)



print('dataset_x_map', dataset_x_map.shape)
print("file list complete^^")
# exit()

def train_binary_only(model):
    print('Layer is Frozen!')

    for l in model.layers:
        # print(l.name.split("_")[len(l.name.split("_")) - 1])
        if l.name.split("_")[len(l.name.split("_")) - 1] == "EN":
            # print(l.name.split("_")[len(l.name.split("_")) - 1])
            l.trainable = False


def main(args):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


    model = SIFTAttNet_v4_encoder_CBAM(blocks=5, input_channel=1, droppath_rate=0.2, dualConv=True, CBAM=True, lastCBAM=False,
                                        var = 0.03, non_linear=True, dilated = True, 
                                        pool='DownSamplingModule', upsample='UpSamplingModule')





    if args.freeze:
        train_binary_only(model)
    train(model=model, epochs=epochs, batch_size=batch_size, status=args.status)


decay = initial_learning_rate / epochs


def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)


def scheduler(epoch, lr):
    if epoch > 10:
        return lr * tf.math.exp(-0.1)
    else:
        return lr


def train(model=None, epochs=100, batch_size=32, status=True):

    ''''''
    sample_count = 30
    total_steps = int(epochs * len(path_y_train) / batch_size)
    warmup_epoch = 10
    warmup_steps = int(warmup_epoch * len(path_y_train) / batch_size)
    print('total_steps', total_steps)
    print('warmup_steps', warmup_steps)

    # exit()

    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=initial_learning_rate,
                                            total_steps=total_steps,
                                            warmup_learning_rate=initial_learning_rate, # 0.0,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=0)
    print('warm_up_lr', warm_up_lr)
    

    # decay_steps = 1000
    # # initial_learning_rate = 0
    # warmup_steps = 1000
    # target_learning_rate = 0.1
    # lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
    #     warmup_steps=warmup_steps
    # )
    # print('lr_warmup_decayed_fn', lr_warmup_decayed_fn)
    
    LossLearningRateScheduler_ = LossLearningRateScheduler(model, initial_learning_rate, lr_csv_path,
                                                           decay_threshold=0.005, decay_rate=0.95, loss_type='loss')
    
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, r"{epoch}-{loss:.2f}.h5"),
                                                    save_best_only=True, monitor="val_loss", save_weights_only=True, ), \
                tf.keras.callbacks.CSVLogger(os.path.join(save_path, "history.csv")), \
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min',baseline=None, restore_best_weights=False),\
                #  warm_up_lr]
                 # LossLearningRateScheduler_]
                 LearningRateScheduler(scheduler)]

    print('============================')
    print('start training...')

    # 原本 0.001 decay: 0.0001
    adam = keras.optimizers.Adam(learning_rate=initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None,
                                 decay=1e-7)

    print(dataset_xtrain.shape)
    print(dataset_ytrain.shape)

    print(dataset_xval.shape)
    print(dataset_yval.shape)

    if status == 'train_wo_map':
        ### 1 in 1 out ###
        print("status == train_wo_map")

        model.compile(optimizer=adam, loss=MSSSIMLossFit)

        history = model.fit(dataset_xtrain,
                            dataset_ytrain,
                            validation_data=(dataset_xval, dataset_yval),
                            batch_size=batch_size, callbacks=callbacks,
                            shuffle=True, epochs=epochs, verbose=1)

    elif status == 'train_w_map':
        print("status == train_w_map")
        model.compile(optimizer=adam, loss=MSSSIMLossFit)

        history = model.fit([dataset_xtrain_single, dataset_x_map],
                            dataset_ytrain_single,
                            validation_data=([dataset_xval_single, dataset_y_map], dataset_yval_single),
                            batch_size=batch_size, callbacks=callbacks,
                            shuffle=True, epochs=epochs, verbose=1)
    
  
    elif status == 'pretrain_w_map':
        print("status == pretrain_w_map")

        print("dataset_xtrain", dataset_xtrain.shape)
        print("dataset_ytrain_single", dataset_ytrain_single.shape)
        print("dataset_x_map", dataset_x_map.shape)
        print("dataset_xval", dataset_xval.shape)
        print("dataset_yval_single", dataset_yval_single.shape)
        print("dataset_y_map", dataset_y_map.shape)
        # exit()

        model.compile(optimizer=adam, loss=MSSSIMLossFit)

        history = model.fit([dataset_xtrain, dataset_x_map],
                            dataset_ytrain,
                            validation_data=([dataset_xval, dataset_y_map], dataset_yval),
                            batch_size=batch_size, callbacks=callbacks,
                            shuffle=True, epochs=epochs, verbose=1)

    matplotlib.use('AGG')  # 或者PDF, SVG或PS

    figurePath = './figure/'
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig(figurePath + 'loss_' + save_name + '.png')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', choices=['0', '1'])
    parser.add_argument('--status', type=str, default='train_w_map',
                        choices=['train_w_map', 'train_wo_map', 'pretrain_w_map'])
    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--channel', type=int, default=1)

    args = parser.parse_args()
    print(args.status)
    main(args)
    print('*****************************')
    print('finish!')
