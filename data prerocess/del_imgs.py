import cv2
import os
import numpy as np

file_path =  "/data/FocalTech/nasic9395_0606_aug_v3_siftMap/val/SiftMap"
del_path1 = "/data/FocalTech/nasic9395_0606_aug_v3_siftMap/val/SiftMap_same"
# del_path2 = './FT9395_20220421_preprocessed/noised'

found = False
for del_img_filename in os.listdir(del_path1):
    found = False
    for img_filename in os.listdir(file_path):
        if(del_img_filename[2:] == img_filename[2:]):
            found = True
            break
    
    if(found == False):
        os.remove(del_path1 + "/" + del_img_filename)
        print("del_img_filename[2:] = ", del_img_filename[2:])


# found = False
# for del_img_filename in os.listdir(del_path2):
#     found = False
#     for img_filename in os.listdir(file_path):
#         if(del_img_filename[2:] == img_filename[2:]):
#             found = True
#             break
    
#     if(found == False):
#         os.remove(del_path2 + "/" + del_img_filename)
#         print("del_img_filename[2:] = ", del_img_filename[2:])