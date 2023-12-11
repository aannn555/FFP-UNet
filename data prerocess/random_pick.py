import time
import cv2
import os
import numpy as np
import math
import shutil

noised_path = "./nasic9395_0613_del_heavy_noise_aug_v2/x_val"
normal_path = "./nasic9395_0613_del_heavy_noise_aug_v2/y_val"

pick_noised = "./nasic9395_0613_del_heavy_noise_aug_v4/x_val"
pick_normal = "./nasic9395_0613_del_heavy_noise_aug_v4/y_val"

os.makedirs(pick_normal, exist_ok=True)
os.makedirs(pick_noised, exist_ok=True)

y_train_cnt = 0
counter = 0

files = os.listdir(normal_path)
num_imgs = len(files)
if '.DS_Store' in files:
    files.remove('.DS_Store')

start_time = time.time()

print("num_imgs = ", num_imgs)
for img_file in files:
    img_all = cv2.imread(normal_path + "/" + img_file, cv2.IMREAD_GRAYSCALE)
    print("counter", counter)
    # print("img_all.shape = ", img_all.shape)
    # print("img_all.dtype = ", img_all.dtype)

    if counter % 10 ==5:
        y_train_cnt = y_train_cnt + 1
        _normal = os.path.join(normal_path, img_file)
        save_normal = os.path.join(pick_normal, img_file)
        shutil.copyfile(_normal, save_normal)

        _noised = os.path.join(noised_path, img_file)
        save_noised = os.path.join(pick_noised, img_file)
        shutil.copyfile(_noised, save_noised)
    else:
        pass

    counter += 1
print('*' * 20)
print("y_train_cnt:", y_train_cnt)
end_time = time.time()
print("Total Spend time：", str((end_time - start_time) / 60)[0:6] + "分鐘")
print('Finish!')