import time
import cv2
import matplotlib.pyplot as plt
from math import cos, sin, radians, fabs
import math
import numpy as np
import random
import os

## https://stackoverflow.com/questions/67333243/how-to-remove-an-image-that-contains-more-than-90-black-pixels


# size = 36
# angle = 60
path = f"./nasic9395_0606_align"
crop_path = f"./nasic9395_0606_align_crop"
save_path = f'./nasic9395_0606_aug_v2'
files = os.listdir(path)

os.makedirs(f'{crop_path}/', exist_ok=True)
os.makedirs(f'{save_path}/', exist_ok=True)

if '.DS_Store' in files:
    files.remove('.DS_Store')


def cropLeftRight(img, image_name):
    height, width = img.shape
    # image_name = file.split(".")[0]
    left = img[0:height, 0:36]
    right = img[0:height, 38:width - 3]
    cv2.imwrite(crop_path + "/" + image_name + '-1.bmp', left)
    cv2.imwrite(crop_path + "/" + image_name + '-2.bmp', right)
    return left, right


def cropHeight(img, img2):
    height, width = img.shape
    ret, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    contours1, hierarchy1 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_start = sorted(np.squeeze(contours1[0], axis=1), key=lambda k: [k[1], k[0]])[0][1]
    h_end = contours1[0].max()

    ret2, thresh2 = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
    contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_start2 = sorted(np.squeeze(contours2[0], axis=1), key=lambda k: [k[1], k[0]])[0][1]
    h_end2 = contours2[0].max()
    if h_end2 - h_start2 > h_end - h_start:
        out_contours = contours2
        if h_start == h_end:
            fingerprint_l = img[0:height, 0:36]
            fingerprint_r = img2[0:height, 0:36]
        else:
            fingerprint_l = img[h_start2:h_end2, 0:36]
            fingerprint_r = img2[h_start2:h_end2, 0:36]

    else:
        out_contours = contours1
        if h_start == h_end:
            fingerprint_l = img[0:height, 0:36]
            fingerprint_r = img2[0:height, 0:36]
        else:
            fingerprint_l = img[h_start:h_end, 0:36]
            fingerprint_r = img2[h_start:h_end, 0:36]

    return fingerprint_l, fingerprint_r, out_contours


def getRotationMatrix2D(theta, cx=0, cy=0):
    # http://www.1zlab.com/wiki/python-opencv-tutorial/opencv-rotation/
    # 角度值转换为弧度值
    # 因为图像的左上角是原点 需要×-1
    theta = math.radians(-1 * theta)

    M = np.float32([
        [math.cos(theta), -math.sin(theta), (1 - math.cos(theta)) * cx + math.sin(theta) * cy],
        [math.sin(theta), math.cos(theta), -math.sin(theta) * cx + (1 - math.cos(theta)) * cy]])
    return M


def rotateImage(angle, fingerprint_l):
    M = getRotationMatrix2D(angle, cx=cX, cy=cY)
    fingerprint_h, fingerprint_w = fingerprint_l.shape
    new_H = int(fingerprint_w * fabs(sin(radians(angle))) + fingerprint_h * fabs(cos(radians(angle))))
    new_W = int(fingerprint_h * fabs(sin(radians(angle))) + fingerprint_w * fabs(cos(radians(angle))))
    offset_w = (new_W - fingerprint_w) / 2
    offset_h = (new_H - fingerprint_h) / 2
    M[0, 2] += offset_w
    M[1, 2] += offset_h
    rotate_img = cv2.warpAffine(fingerprint_l, M, (new_W, new_H), borderValue=(0, 0, 0))
    return rotate_img


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def splitData(X_points, Y_points, img, filename, size, angle, save_folder):
    count = 0
    name = filename
    frmt = 'bmp'

    for i in Y_points:
        for j in X_points:
            black = 0
            percent = 0
            split = img[i:i + size, j:j + size]
            for y in range(size):
                for x in range(size):
                    if split.shape == (36, 36):
                        pixel = split[y, x]
                        if pixel == 0:
                            black += 1
            # print(black)
            percent = black / (size * size)
            if percent < 0.1:
                os.makedirs(f'./{save_path}/{save_folder}/', exist_ok=True)
                patch = cv2.imwrite(
                    f'./{save_path}/{save_folder}/' + '{}_{}_{}.{}'.format(name, angle, count, frmt),
                    split)

            # print(percent)
            count += 1
    # return patch


if __name__ == '__main__':
    start_time = time.time()
    angle_range = np.arange(-180, 180 + 1, 45)
    for i in range(angle_range.size-1):
        angles = []
        angles = random.sample(range(angle_range[i], angle_range[i + 1]), 5)
        # degree_file = f'{angle_range[i]}_{angle_range[i + 1]}/'
        sizes = [36]
        j = 0
        for file in files:
            img = cv2.imread(path + "/" + file, cv2.IMREAD_GRAYSCALE)
            image_name = file.split(".")[0]
            height, width = img.shape
            print("No.:", j, " =========================================================")
            print("File:", file)

            filename_l = image_name + '-1.bmp'
            filename_r = image_name + '-2.bmp'

            ### Divide into Left-Right ###
            left, right = cropLeftRight(img, image_name)

            ### Crop Height, get Contours ###
            fingerprint_l, fingerprint_r, contours_l = cropHeight(left, right)
            # cv2.imwrite('./crop/' + filename_l, fingerprint_l)
            # cv2.imwrite('./crop/' + filename_r, fingerprint_r)

            ### Get Moments ###
            cnt = contours_l[0]
            M = cv2.moments(cnt)
            if M['m10'] == 0.0 and M['m00'] == 0.0:
                cx = width // 2
                cy = height // 2
            else:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
            for angle in angles:
                print(f'*** i = {i} / angle = {angle} ***')
                for size in sizes:
                    # print(f'*** size = {size} ***')
                    rotate_l = rotateImage(angle, fingerprint_l)
                    rotate_r = rotateImage(angle, fingerprint_r)
                    # cv2.imwrite('./samp_rotate/' + filename_l, rotate_l)
                    # cv2.imwrite('./samp_rotate/' + filename_r, rotate_r)

                    fingerprint_h, fingerprint_w = rotate_l.shape

                    X_points = start_points(fingerprint_w, size, 0.5)
                    Y_points = start_points(fingerprint_h, size, 0.5)

                    save_folder_normal = 'normal'
                    save_folder_blur = 'noised'

                    splitData(X_points, Y_points, rotate_l, image_name, size, angle, save_folder_normal)
                    splitData(X_points, Y_points, rotate_r, image_name, size, angle, save_folder_blur)
            j = j + 1
    print('Finish!')

    # print(file)
    end_time = time.time()
    print("Total Spend time：", str((end_time - start_time) / 60)[0:6] + "分鐘")
