import cv2
import os
import numpy as np
import math
import shutil


# b_normal_path = "./nasic9395_0606_aug_v3/binary/clear"
# nb_normal_path = "./nasic9395_0606_aug_v3/non_binary/clear"
# nb_noised_path = "./nasic9395_0606_aug_v3/non_binary/noise"

b_normal_path = "/data/FocalTech/nasic9395_0116_unzip/y_train_binary/"
b_noised_path = "/data/FocalTech/nasic9395_0116_unzip/x_train_binary/"
nb_normal_path = "/data/FocalTech/nasic9395_0116_unzip/y_train/"
nb_noised_path = "/data/FocalTech/nasic9395_0116_unzip/x_train/"

# testset_b_normal = "/data/FocalTech/nasic9395_0116_unzip/testset/binary/y_test"
# testset_nb_normal = "/data/FocalTech/nasic9395_0116_unzip/testset/non_binary/y_test"
# testset_b_noised = "/data/FocalTech/nasic9395_0116_unzip/testset/binary/x_test"
# testset_nb_noised = "/data/FocalTech/nasic9395_0116_unzip/testset/non_binary/x_test"

train_b_normal = "/data/FocalTech/nasic9395_0116_unzip/train/binary/y_train"
train_nb_normal = "/data/FocalTech/nasic9395_0116_unzip/train/non_binary/y_train"
train_b_noised = "/data/FocalTech/nasic9395_0116_unzip/train/binary/x_train"
train_nb_noised = "/data/FocalTech/nasic9395_0116_unzip/train/non_binary/x_train"

val_b_normal = "/data/FocalTech/nasic9395_0116_unzip/val/binary/y_val"
val_nb_normal = "/data/FocalTech/nasic9395_0116_unzip/val/non_binary/y_val"
val_b_noised = "/data/FocalTech/nasic9395_0116_unzip/val/binary/x_val"
val_nb_noised = "/data/FocalTech/nasic9395_0116_unzip/val/non_binary/x_val"

img_size_h = 176
img_size_w = 36

# os.makedirs(testset_b_normal, exist_ok=True)
# os.makedirs(testset_nb_normal, exist_ok=True)
# os.makedirs(testset_b_noised, exist_ok=True)
# os.makedirs(testset_nb_noised, exist_ok=True)

os.makedirs(train_b_normal, exist_ok=True)
os.makedirs(train_nb_normal, exist_ok=True)
os.makedirs(train_b_noised, exist_ok=True)
os.makedirs(train_nb_noised, exist_ok=True)


os.makedirs(val_b_normal, exist_ok=True)
os.makedirs(val_nb_normal, exist_ok=True)
os.makedirs(val_b_noised, exist_ok=True)
os.makedirs(val_nb_noised, exist_ok=True)


#########################################
# array size determination
#########################################
y_train_cnt = 0
y_val_cnt = 0
counter = 0

num_imgs = len(os.listdir(b_normal_path))
print("num_imgs = ", num_imgs)
for img_file in os.listdir(b_normal_path):
    print(b_normal_path + "/" + img_file)
    img_all = cv2.imread(b_normal_path + "/" + img_file, cv2.IMREAD_GRAYSCALE)
    print("img_all.shape = ", img_all.shape)
    print("img_all.dtype = ", img_all.dtype)

    # TRAIN #
    if(counter % 10 <= 7):
        img_old_path = b_normal_path + "/" + img_file
        img_new_path = train_b_normal + "/" + img_file
        shutil.copyfile(img_old_path, img_new_path)

        img_old_path = b_noised_path + "/" + img_file
        img_new_path = train_b_noised + "/" + img_file
        shutil.copyfile(img_old_path, img_new_path)

        img_old_path = nb_normal_path + "/" + img_file
        img_new_path = train_nb_normal + "/" + img_file
        shutil.copyfile(img_old_path, img_new_path)

        img_old_path = nb_noised_path + "/" + img_file
        img_new_path = train_nb_noised + "/" + img_file
        shutil.copyfile(img_old_path, img_new_path)

        y_train_cnt = y_train_cnt + 1

    # VAL #
    # elif(counter % 10 == 8):
    else:
        y_val_cnt = y_val_cnt + 1

        img_old_path = b_normal_path + "/" + img_file
        img_new_path = val_b_normal + "/" + img_file
        shutil.copyfile(img_old_path, img_new_path)

        img_old_path = b_noised_path + "/" + img_file
        img_new_path = val_b_noised + "/" + img_file
        shutil.copyfile(img_old_path, img_new_path)

        img_old_path = nb_normal_path + "/" + img_file
        img_new_path = val_nb_normal + "/" + img_file
        shutil.copyfile(img_old_path, img_new_path)

        img_old_path = nb_noised_path + "/" + img_file
        img_new_path = val_nb_noised + "/" + img_file
        shutil.copyfile(img_old_path, img_new_path)


    # else:
    #     img_old_path = b_normal_path + "/" + img_file
    #     img_new_path = testset_b_normal + "/" + img_file
    #     shutil.copyfile(img_old_path, img_new_path)

    #     img_old_path = b_noised_path + "/" + img_file
    #     img_new_path = testset_b_noised + "/" + img_file
    #     shutil.copyfile(img_old_path, img_new_path)

    #     img_old_path = nb_normal_path + "/" + img_file
    #     img_new_path = testset_nb_normal + "/" + img_file
    #     shutil.copyfile(img_old_path, img_new_path)

    #     img_old_path = nb_noised_path + "/" + img_file
    #     img_new_path = testset_nb_noised + "/" + img_file
    #     shutil.copyfile(img_old_path, img_new_path)

    counter += 1

x_train_cnt = 0
x_val_cnt = 0
counter = 0

num_imgs = len(os.listdir(nb_noised_path))
print("num_imgs = ", num_imgs)
for img_file in os.listdir(nb_noised_path):
    img_all = cv2.imread(nb_noised_path + "/" + img_file, cv2.IMREAD_GRAYSCALE)
    print("img_all.shape = ", img_all.shape)
    print("img_all.dtype = ", img_all.dtype)

    if(counter % 10 <= 7):
        x_train_cnt = x_train_cnt + 1
    # elif(counter % 10 == 8):
    #     x_val_cnt = x_val_cnt + 1
    else:
        x_val_cnt = x_val_cnt + 1
        # pass

    counter += 1


print("y_train_cnt = ", y_train_cnt)
print("y_val_cnt = ", y_val_cnt)
print("x_train_cnt = ", x_train_cnt)
print("x_val_cnt = ", x_val_cnt)


#########################################
# array construction
#########################################

y_train = np.empty([y_train_cnt, img_size_h, img_size_w, 2], dtype=np.uint8)
y_val = np.empty([y_val_cnt, img_size_h, img_size_w, 2], dtype=np.uint8)
x_train = np.empty([x_train_cnt, img_size_h, img_size_w, 2], dtype=np.uint8)
x_val = np.empty([x_val_cnt, img_size_h, img_size_w, 2], dtype=np.uint8)

y_train_cnt = 0
y_val_cnt = 0
counter = 0

num_imgs = len(os.listdir(b_normal_path))
print("num_imgs = ", num_imgs)
for img_file in os.listdir(b_normal_path):
    img_all_b = cv2.imread(b_normal_path + "/" + img_file, cv2.IMREAD_GRAYSCALE)
    img_all_nb = cv2.imread(nb_normal_path + "/" + img_file, cv2.IMREAD_GRAYSCALE)
    

    img_b = img_all_b
    img_nb = img_all_nb
    img_b = img_b.reshape((img_b.shape[0], img_b.shape[1], 1))
    img_nb = img_nb.reshape((img_nb.shape[0], img_nb.shape[1], 1))

    append_img = np.append(img_nb, img_b, axis=2)
    print("append_img.shape = ", append_img.shape)
    
    if(counter % 10 <= 7):
        y_train[y_train_cnt] = append_img
        y_train_cnt = y_train_cnt + 1
    elif(counter % 10 == 8):
        y_val[y_val_cnt] = append_img
        y_val_cnt = y_val_cnt + 1
    else:
        pass
    

    counter += 1

x_train_cnt = 0
x_val_cnt = 0
counter = 0

num_imgs = len(os.listdir(nb_noised_path))
print("num_imgs = ", num_imgs)
for img_file in os.listdir(nb_noised_path):
    img_all_b = cv2.imread(nb_noised_path + "/" + img_file, cv2.IMREAD_GRAYSCALE)
    img_all_nb = cv2.imread(nb_noised_path + "/" + img_file, cv2.IMREAD_GRAYSCALE)
    

    img_b = img_all_b
    img_nb = img_all_nb
    img_b = img_b.reshape((img_b.shape[0], img_b.shape[1], 1))
    img_nb = img_nb.reshape((img_nb.shape[0], img_nb.shape[1], 1))
    append_img = np.append(img_nb, img_b, axis=2)
    print("append_img.shape = ", append_img.shape)


    # if append_img.shape != (img_size_h,img_size_w,2):
    #     print('Fail')
    #     break

    if(counter % 10 <= 7):
        x_train[x_train_cnt] = append_img
        x_train_cnt = x_train_cnt + 1

    else:
        x_val[x_val_cnt] = append_img
        x_val_cnt = x_val_cnt + 1
    # else:
    #     pass

    counter += 1


print("y_train.shape[0] = %d, y_train_cnt = %d" %(y_train.shape[0], y_train_cnt))
print("y_val.shape[0] = %d, y_val = %d" %(y_val.shape[0], y_val_cnt))
print("x_train.shape[0] = %d, x_train_cnt = %d" %(x_train.shape[0], x_train_cnt))
print("x_val.shape[0] = %d, x_val = %d" %(x_val.shape[0], x_val_cnt))

print("y_train.shape = ", y_train.shape)
print("y_val.shape = ", y_val.shape)
print("x_train.shape = ", x_train.shape)
print("x_val.shape = ", x_val.shape)


# np.savez_compressed('nasic9395_0606_aug_v3.npz', x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)