import tensorflow as tf
import cv2
import os
import numpy as np
from FPNResUNet_baseline import *
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

SAVE_NAME = r"0804_MimoUNet_v5_sample_v1_origin_single_input/"
epoch = 21

model = MimoUNet_v5_sample_v1(blocks=5, input_channel=2)

model.load_weights("/home/han/Anting/MimoUNet0303/0913_MimoUNet_v5_sample_v1_block5_nasic9395_0606_aug_v3_2nd_BEST!/46-0.28.h5")

mode = 'RECOGN_on_testsetv9'
# RECOGN_on_testsetv9 // ENROLL_on_testsetv8

if mode == 'RECOGN':
    ### test_result
    SAVE_ROOT = r"./recogition/"
    PATH = r"/home/han/Anting/dataset/identify/"
    BIN_PATH = r"/home/han/Anting/dataset/identify_binary/"

    X_PATH = sorted(glob('/home/han/Anting/dataset/identify/*.bmp'))
    X_BIN_PATH = sorted(glob('/home/han/Anting/dataset/identify_binary/*.bmp'))

    # PATH = '/data/FocalTech/display_fingerprint/ft_lightnoised/x_test/non_binary/'
    # BIN_PATH = '/data/FocalTech/display_fingerprint/ft_lightnoised/x_test/binary/'

    # X_PATH = sorted(glob('/data/FocalTech/display_fingerprint/ft_lightnoised/x_test/non_binary/*.bmp'))
    # X_BIN_PATH = sorted(glob('/data/FocalTech/display_fingerprint/ft_lightnoised/x_test/binary/*.bmp'))

elif mode == 'ENROLL_on_testsetv9':
    height = 184
    SAVE_ROOT = r"./enroll_on_nasicTestsetv9/"

    PATH = r"/data/FocalTech/testset_nasic9395_v9/enroll/"
    BIN_PATH = r"/data/FocalTech/testset_nasic9395_v9/enroll_binary/"

    X_PATH = sorted(glob(fr'{PATH}*.bmp'))
    X_BIN_PATH = sorted(glob(fr'{BIN_PATH}*.bmp'))

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
SAVE_PATH_BIN = SAVE_ROOT+SAVE_NAME+r'binary'+fr'{epoch}'

NORM_SAVE_PATH = SAVE_ROOT+SAVE_NAME+r'NORM_non_binary'
NORM_SAVE_PATH_BIN = SAVE_ROOT+SAVE_NAME+r'NORM_binary'

normal = lambda image: np.array(image/255, dtype=np.float32)
denormal = lambda image: np.array(image*255, dtype=np.uint8)

print('SAVE_PATH',SAVE_PATH)

### contrastive loss ###
# denormal = lambda image: np.array(255-(image*255), dtype=np.uint8)


def padding_image(img): 

    padding_img = np.full((184, 40), 0) 
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
                img.shape[1] - pad_w:2 * img.shape[1] + (pad_w+1)]
    elif(dif_h % 2 == 1) & (dif_w % 2 == 1):
        cropArea = fullPadding[img.shape[0] - pad_h:2 * img.shape[0] + (pad_h + 1),
                img.shape[1] - pad_w:2 * img.shape[1] + (pad_w+1)]
    else:
        cropArea = fullPadding[img.shape[0] - pad_h:2 * img.shape[0] + pad_h,
                img.shape[1] - pad_w:2 * img.shape[1] + pad_w]


    if cropArea.shape != (184, 40):
        print("fail anting")
        print(filename)
        print(img.shape)
        print("*"*50)
        print(cropArea)
        print(cropArea.shape)
        # exit()
        return False
    else:
        # array_of_img.append(cropArea)
        return cropArea


array_image=[]
array_bin_image=[]

filenames = [os.path.basename(x) for x in (X_PATH)]
_filenames = [os.path.basename(x) for x in (X_BIN_PATH)]

print(len(filenames))
print(len(_filenames))

# exit()

path_num = len(X_PATH)
needed_multi_channel_img = np.zeros((path_num, 184, 40, 2))
i=0

for filename in filenames:
    # print(filename)
    path = os.path.join(PATH, filename)
    bin_path = os.path.join(BIN_PATH, filename)

    image = cv2.imread(path,0)
    bin_image = cv2.imread(bin_path,0)

    image_padding = padding_image(image)
    bin_image_padding = padding_image(bin_image)

    norm_image = normal(image_padding)
    norm_bin_image = normal(bin_image_padding)

    needed_multi_channel_img[i,:,:,0] = norm_image
    needed_multi_channel_img[i,:,:,1] = norm_bin_image

    i = i+1
    if norm_image is not False and norm_bin_image is not False:
        array_image.append(norm_image)
        array_bin_image.append(norm_bin_image)
    
prediction = model.predict(needed_multi_channel_img)


print(path)
print(bin_image.shape)
print('#'*20)

for m in range(path_num):
    path = os.path.join(PATH, filenames[m])
    bin_path = os.path.join(BIN_PATH, filenames[m])

    image = cv2.imread(path,0)
    bin_image = cv2.imread(bin_path,0)

    output = prediction[m,:,:,0]
    output_bin = prediction[m,:,:,1]

    pad1 = (184-image.shape[0])//2
    pad2 = (40-image.shape[1])//2

    output_crop = output[pad1:image.shape[0]+pad1,pad2:image.shape[1]+pad2]
    output_bin_crop = output_bin[pad1:image.shape[0]+pad1,pad2:image.shape[1]+pad2]

    output = denormal(output_crop)
    output_bin = denormal(output_bin_crop)

    os.makedirs(os.path.join(SAVE_PATH), exist_ok=True)
    cv2.imwrite(os.path.join(SAVE_PATH,filenames[m]), output)

    os.makedirs(os.path.join(SAVE_PATH_BIN), exist_ok=True)
    cv2.imwrite(os.path.join(SAVE_PATH_BIN,filenames[m]), output_bin)



print("test pass ^^")


