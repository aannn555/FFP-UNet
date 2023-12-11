from asyncio import FastChildWatcher
from doctest import debug
from re import T
import tensorflow as tf
import cv2
import os
import numpy as np

from FPNResUNet import *


from glob import glob

SAVE_NAME = r"0822_SIFTAttNet_v4_bottleneck_CBAM/"
epoch = 27

# model = MimoUNet_v5_sample_v1(blocks=5 ,input_channel=2)
model = SIFTAttNet_v4_encoder(blocks=5, input_channel=1, FAM_nodule = False, droppath_rate=0, lastCBAM=False)
# model = SIFTAttNet_v4_encoder_CBAM(blocks=5, input_channel=1, droppath_rate=0.2, dualConv=True, CBAM=True, lastCBAM=False,
#                                     var = 0.03, non_linear=True, dilated = True,
#                                     pool='DownSamplingModule', upsample='UpSamplingModule')


notGaussian = False

model.load_weights("/home/han/Anting/FPN_ResUNet/checkpoint/FPNResUNet_SimpleFModule/25-0.22.h5")
inputImage = 2  # 如果包含map=2 不包含map=1
mode = 'RECOGN_on_testsetv9'


if mode == 'RECOGN':
    ### test_result43
    SAVE_ROOT = r"./recognition/"
    height = 184

    PATH = r"/data/FocalTech/nasic9395_0606_aug_v3_siftMap/fullsize_test/non_binary/"
    BIN_PATH = r"/data/FocalTech/nasic9395_0606_aug_v3_siftMap/fullsize_test/binary/"
    MAP_PATH = r"/data/FocalTech/nasic9395_0606_aug_v3_siftMap/fullsize_test/Gaussian_SIFT/"

    X_PATH = sorted(glob('/data/FocalTech/nasic9395_0606_aug_v3_siftMap/fullsize_test/non_binary/*.bmp'))
    X_BIN_PATH = sorted(glob('/data/FocalTech/nasic9395_0606_aug_v3_siftMap/fullsize_test/binary/*.bmp'))
    X_MAP_PATH = sorted(glob('/data/FocalTech/nasic9395_0606_aug_v3_siftMap/fullsize_test/Gaussian_SIFT/*.bmp'))

elif mode == 'RECOGN_testtool_feature_points':
    ### test_result
    height = 184
    SAVE_ROOT = r"./recognition_testtool_feature_points/"

    PATH = r'/data/FocalTech/testsetv7_diff_testtool_feature_points/identify/'
    BIN_PATH = r"/data/FocalTech/testsetv7_diff_testtool_feature_points/identify_binary/"
    MAP_PATH = r"/data/FocalTech/testsetv7_diff_testtool_feature_points/siftMap/"

    X_PATH = sorted(glob('/data/FocalTech/testsetv7_diff_testtool_feature_points/identify/*.bmp'))
    X_BIN_PATH = sorted(glob('/data/FocalTech/testsetv7_diff_testtool_feature_points/identify_binary/*.bmp'))
    X_MAP_PATH = sorted(
        glob('/data/FocalTech/testsetv7_diff_testtool_feature_points/siftMap/*.bmp'))

elif mode == 'RECOGN_on_testsetv8':
    height = 184
    SAVE_ROOT = r"./recognition_on_nasicTestsetv8/"

    PATH = r"/data/FocalTech/testset_nasic9395_v8/noBinary/identify/"
    BIN_PATH = r"/data/FocalTech/testset_nasic9395_v8/withBinary/identify_binary/"
    MAP_PATH = r"/data/FocalTech/testset_nasic9395_v8/noBinary/siftMap/"

    X_PATH = sorted(glob(fr'{PATH}*.bmp'))
    X_BIN_PATH = sorted(glob(fr'{BIN_PATH}*.bmp'))
    X_MAP_PATH = sorted(glob(fr'{MAP_PATH}*.bmp'))
elif mode == 'RECOGN_on_testsetv9':
    height = 184
    SAVE_ROOT = r"./recognition_on_nasicTestsetv9/"

    PATH = r"/data/FocalTech/testset_nasic9395_v9/identify/"
    BIN_PATH = r"/data/FocalTech/testset_nasic9395_v9/identify_binary/"
    MAP_PATH = r"/data/FocalTech/testset_nasic9395_v9/siftMap/"

    X_PATH = sorted(glob(fr'{PATH}*.bmp'))
    X_BIN_PATH = sorted(glob(fr'{BIN_PATH}*.bmp'))
    X_MAP_PATH = sorted(glob(fr'{MAP_PATH}*.bmp'))

SAVE_PATH = SAVE_ROOT + SAVE_NAME + r'non_binary_'+fr'{epoch}'
SAVE_PATH_BIN = SAVE_ROOT + SAVE_NAME + r'binary'

print('Model:', mode)
print(len(X_PATH))
print(len(X_BIN_PATH))
print(len(X_MAP_PATH))

normal = lambda image: np.array(image / 255, dtype=np.float32)
denormal = lambda image: np.array(image * 255, dtype=np.uint8)


array_image = []
array_bin_image = []

filenames = [os.path.basename(x) for x in X_PATH]
print(len(filenames))

path_num = len(X_PATH)

def convertMap(map):
    map[map > 0] = 1
    return map


def padding_image(img, height):
    padding_img = np.full((height, 40), 0)
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
    else:
        cropArea = fullPadding[img.shape[0] - pad_h:2 * img.shape[0] + pad_h,
                   img.shape[1] - pad_w:2 * img.shape[1] + pad_w]

    if cropArea.shape != (height, 40):
        print("fail")
        print(img.shape)
        print("*" * 50)
        return False
    else:
        return cropArea



if inputImage == 2:
    needed_multi_channel_img = np.zeros((path_num, height, 40, 1))
    map_list = np.zeros((path_num, height, 40, 1))
    i = 0
    for filename in filenames:
        # print(filename)
        path = os.path.join(PATH, filename)
        map_path = os.path.join(MAP_PATH, filename)

        image = cv2.imread(path, 0)
        map_image = cv2.imread(map_path, 0)

        image_padding = padding_image(image, height = height)
        map_image_padding = padding_image(map_image, height = height)

        

        norm_image = normal(image_padding)
        map_norm_image = normal(map_image_padding)

        if notGaussian == True:
            print('notGaussian '*5)
            map_norm_image = convertMap(map_norm_image)

        needed_multi_channel_img[i, :, :, 0] = norm_image
        map_list[i, :, :, 0] = map_norm_image

        i = i + 1
        # if norm_image is not False:
        #     array_image.append(norm_image)

    ### 1 output
    prediction = model.predict([needed_multi_channel_img, map_list])

    print(path)
    print('#' * 20)

    ### 1 output
    for m in range(path_num):
        path = os.path.join(PATH, filenames[m])
        image = cv2.imread(path, 0)
        output = prediction[m]

        pad1 = (height - image.shape[0]) // 2
        pad2 = (40 - image.shape[1]) // 2

        output_crop = output[pad1:image.shape[0] + pad1, pad2:image.shape[1] + pad2]
        output = denormal(output_crop)

        os.makedirs(os.path.join(SAVE_PATH), exist_ok=True)
        cv2.imwrite(os.path.join(SAVE_PATH, filenames[m]), output)

elif inputImage == 1:
    needed_multi_channel_img = np.zeros((path_num, height, 40, 1))
    i = 0
    for filename in filenames:
        # print(filename)
        path = os.path.join(PATH, filename)
        image = cv2.imread(path, 0)
        image_padding = padding_image(image, height = height)
        norm_image = normal(image_padding)
        needed_multi_channel_img[i, :, :, 0] = norm_image
        i = i + 1
        if norm_image is not False:
            array_image.append(norm_image)

    ### 1 output
    prediction = model.predict(needed_multi_channel_img)

    print(path)
    print('#' * 20)

    ### 1 output
    for m in range(path_num):
        path = os.path.join(PATH, filenames[m])
        image = cv2.imread(path, 0)
        output = prediction[m]

        pad1 = (height - image.shape[0]) // 2
        pad2 = (40 - image.shape[1]) // 2

        output_crop = output[pad1:image.shape[0] + pad1, pad2:image.shape[1] + pad2]
        output = denormal(output_crop)

        os.makedirs(os.path.join(SAVE_PATH), exist_ok=True)
        cv2.imwrite(os.path.join(SAVE_PATH, filenames[m]), output)

print("test pass ^^")
