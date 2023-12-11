import cv2
import os
import math
import numpy as np
import shutil
import time
import multiprocessing
from multiprocessing import Process, Pool

outside_folder = f'/data/FocalTech/nasic9395_0606_aug_v3_siftMap'
status = 'train'

dst_root = f"/data/FocalTech/nasic9395_0606_aug_v3/{status}/non_binary/y_{status}/"
# dst_root = "/data/FocalTech/nasic9395_0606_aug_v3/aug/"
src_root = f"/data/FocalTech/nasic9395_0606_align_crop/"

nb_x_root=f"/data/FocalTech/nasic9395_0606_aug_v3/{status}/non_binary/x_{status}/"
nb_y_root=f"/data/FocalTech/nasic9395_0606_aug_v3/{status}/non_binary/y_{status}/"
b_x_root=f"/data/FocalTech/nasic9395_0606_aug_v3/{status}/binary/x_{status}/"
b_y_root=f"/data/FocalTech/nasic9395_0606_aug_v3/{status}/binary/y_{status}/"

path_list = os.listdir(dst_root)
num_imgs = len(path_list)

if '.DS_Store' in path_list:
    path_list.remove('.DS_Store')
src_path_list = os.listdir(src_root)
if '.DS_Store' in src_path_list:
    src_path_list.remove('.DS_Store')

num_process = multiprocessing.cpu_count()



output_full_folder_root = os.path.join(outside_folder, status, 'SiftMap_all_same', '') # include black 
dst_new_folder_root = os.path.join(outside_folder, status, 'SiftMap_same', '') # without black

output_nb_x_root = os.path.join(outside_folder, status, f'non_binary/x_{status}', '') # train/non_binary/x_train
output_nb_y_root = os.path.join(outside_folder, status, f'non_binary/y_{status}', '') # train/non_binary/y_train
output_b_x_root = os.path.join(outside_folder, status, f'binary/x_{status}', '') # train/binary/x_train
output_b_y_root = os.path.join(outside_folder, status, f'binary/y_{status}', '') # train/binary/y_train


try:
    os.makedirs(dst_new_folder_root, exist_ok=True)
    os.makedirs(output_full_folder_root, exist_ok=True)
    os.makedirs(output_nb_x_root, exist_ok=True)
    os.makedirs(output_nb_y_root, exist_ok=True)
    os.makedirs(output_b_x_root, exist_ok=True)
    os.makedirs(output_b_y_root, exist_ok=True)

    print('Create Success!')

except OSError as error:
    pass


def extractFilename(file):
    token_type = file.split('.')
    fullName = token_type[0]

    token = file.split('_')
    partialName = token[0]
    for i in range(1, 5):
        partialName = "_".join([partialName, token[i]])
        partialName = partialName

    angle = int(token[-2])

    return partialName, fullName, angle


# 定义旋转rotate函数
def rotate(image, angle: float, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(angle=angle, center=center, scale=1)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


# Gaussian filter
def getNode(N):
    start = N // 2
    end = N - start
    return start, end


def GaussianFilter(src, N=35, sigma_s=100):
    """ Gaussian filter

    Args:
        src (float ndarray, shape (height, width)): source intensity
        N (int): window size of the filter (Defaults to 35)
                 filter indices span [-N/2, N/2]
        sigma_s (float): standard deviation of Gaussian filter (Defaults to 100)

    Returns:
        result (float ndarray, shape (height, width))
    """

    ''' TODO '''
    height = src.shape[0]
    width = src.shape[1]
    kernel = np.zeros(shape=(N, N))
    bound1, bound2 = getNode(N)
    filter_dis = N // 2

    # Filter Kernel
    for i in range(-bound1, bound2):
        for j in range(-bound1, bound2):
            distance = (i ** 2 + j ** 2)
            kernel[i + filter_dis, j + filter_dis] = np.exp((-1) * distance / (2 * (sigma_s ** 2)))

    # Base Layer
    kernel_size = kernel.shape[0]
    temp = np.zeros((height, width))
    weight = np.zeros((height, width))
    image = src.copy()
    kernel_center = filter_dis + 1
    gaus_kernel = kernel.copy()
    for i in range(height):
        for j in range(width):
            left = j
            x_min = filter_dis - left if left <= filter_dis else 0

            right = (width - 1) - j
            x_max = (kernel_center + right) if right <= filter_dis else kernel_size

            top = i
            y_min = filter_dis - top if top <= filter_dis else 0

            down = (height - 1) - i
            y_max = (kernel_center + down) if down <= filter_dis else kernel_size

            temp[i, j] = np.sum(gaus_kernel[y_min:y_max, x_min:x_max] *
                                image[i + y_min - filter_dis:i + y_max - filter_dis,
                                j + x_min - filter_dis:j + x_max - filter_dis])
            weight[i, j] = np.sum(gaus_kernel[y_min:y_max, x_min:x_max])

    # result = temp / weight
    result = temp  # / weight

    return result

def sameKernel(map, N):
    height = map.shape[0]
    width = map.shape[1]
    kernel = np.zeros(shape=(N, N))
    bound1, bound2 = getNode(N)
    filter_dis = N // 2

    # Filter Kernel
    for i in range(-bound1, bound2):
        for j in range(-bound1, bound2):
            distance = (i ** 2 + j ** 2)
            kernel[i + filter_dis, j + filter_dis] = 1
    # Base Layer
    kernel_size = kernel.shape[0]
    temp = np.zeros((height, width))
    weight = np.zeros((height, width))
    image = map.copy()
    kernel_center = filter_dis + 1
    gaus_kernel = kernel.copy()

    for i in range(height):
        for j in range(width):
            left = j
            x_min = filter_dis - left if left <= filter_dis else 0

            right = (width - 1) - j
            x_max = (kernel_center + right) if right <= filter_dis else kernel_size

            top = i
            y_min = filter_dis - top if top <= filter_dis else 0

            down = (height - 1) - i
            y_max = (kernel_center + down) if down <= filter_dis else kernel_size
            temp[i, j] = np.sum(gaus_kernel[y_min:y_max, x_min:x_max] *
                                image[i + y_min - filter_dis:i + y_max - filter_dis,
                                j + x_min - filter_dis:j + x_max - filter_dis])

    result = temp  # / weight

    return result


def drawBlankMap(src_pts, dst_pts):
    _src_pts = src_pts.reshape(-1, 2)
    _dst_pts = dst_pts.reshape(-1, 2)
    img = np.zeros((36, 36, 3), np.uint8)

    for point in _dst_pts:
        img[int(point[1])][int(point[0])] = (255, 255, 255)
    SIFT_map = img

    return SIFT_map


def drawGaussian(map, kernel=5, std=0.707, Gaussian=True):
    blur = np.zeros(map.shape, dtype=np.float32)
    for i in range(3):
        if Gaussian:
            blur[:, :, i] = GaussianFilter(map[:, :, i], N=kernel, sigma_s=std)
        else:
            blur[:, :, i] = sameKernel(map[:, :, i], N=kernel)

    Gaus_SIFT_map = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

    return Gaus_SIFT_map


def drawGaussianSIFTMap(src_pts, dst_pts, kernel=7, std=1.25, Gaussian=True):
    SIFT_map = drawBlankMap(src_pts, dst_pts)
    Gaus_SIFT_map = drawGaussian(SIFT_map, kernel=kernel, std=std, Gaussian=Gaussian)

    return Gaus_SIFT_map


def SIFTwithRANSAC(img1, img2, threshold=0.8, MIN_MATCH_COUNT=5):
    print('START SIFTwithRANSAC')
    global src_pts, check_dst
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=500)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append(m)
        good_matches_num = len(good)
    
    print("good_matches_num", good_matches_num)

    if len(good) > MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        src_pts_size = np.float32([kp1[m.queryIdx].size for m in good]).reshape(-1, 1, 2)
        dst_pts_size = np.float32([kp2[m.trainIdx].size for m in good]).reshape(-1, 1, 2)

        # print('src_pts, dst_pts', src_pts, dst_pts)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        matchesMask = mask.ravel().tolist()

        # Draw detected template in scene image
        h, w = img1.shape
        pts = np.float32([[0, 0],
                          [0, h - 1],
                          [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        good_matches_num = sum(matchesMask)

    else:
        print('No RANSAC')
        matchesMask = None

    # draw_params = dict(singlePointColor=None,
    #                    matchesMask=matchesMask,  # draw only inliers
    #                    flags=2)

    # matches_img_sift = cv2.drawMatches(img1, kp1, img2, kp2,
    #                                    good, None, **draw_params)

    # _src_pts = src_pts.reshape(-1, 2)
    # _dst_pts = dst_pts.reshape(-1, 2)

    # Check Correct Points
    # for point in _src_pts:
    #     check_src = cv2.circle(rgb_img1, (int(point[0]), int(point[1])), 1, (0, 0, 255), -1)
    # for point in _dst_pts:
    #     check_dst = cv2.circle(rgb_img2, (int(point[0]), int(point[1])), 1, (0, 0, 255), -1)

    
    return src_pts, dst_pts


def job(id):
    imgs = path_list[num_imgs * id // num_process: num_imgs * (id + 1) // num_process]
    print(len(imgs))

    for file in imgs:
        threshold = 0.85
        MIN_MATCH_COUNT = 3

        # Preprocess File Name
        print('file', file)
        _partialName, nameWithAngle, angle = extractFilename(file)
        save_full_path = os.path.join(output_full_folder_root, nameWithAngle + ".bmp")
        print("save_full_path", _partialName, nameWithAngle, angle)


        input_fileName = _partialName + '-1.bmp'
        src_img_name = os.path.join(src_root, input_fileName)
        src_img1 = cv2.imread(src_img_name)
        print("src_img_name",src_img_name)
        img1 = cv2.cvtColor(src_img1, cv2.COLOR_BGR2GRAY)

        dst_img_name = os.path.join(dst_root, file)
        img_new_path = os.path.join(dst_new_folder_root, file)


        _nb_x_root = os.path.join(nb_x_root, file)
        _output_nb_x_root = os.path.join(output_nb_x_root, file)

        _nb_y_root = os.path.join(nb_y_root, file)
        _output_nb_y_root = os.path.join(output_nb_y_root, file)

        _b_x_root = os.path.join(b_x_root, file)
        _output_b_x_root = os.path.join(output_b_x_root, file)

        _b_y_root = os.path.join(b_y_root, file)
        _output_b_y_root = os.path.join(output_b_y_root, file)

        print('-'*20)

        # break
        # continue

        try:
            dst_img2 = cv2.imread(dst_img_name)
            img2 = cv2.cvtColor(dst_img2, cv2.COLOR_BGR2GRAY)

            src_pts, dst_pts = SIFTwithRANSAC(img1, img2,
                                              threshold=threshold,
                                              MIN_MATCH_COUNT=MIN_MATCH_COUNT)
            _GaussianSIFTMap = drawGaussianSIFTMap(src_pts, dst_pts, Gaussian=False, kernel=5)

            cv2.imwrite(save_full_path, _GaussianSIFTMap)
            cv2.imwrite(img_new_path, _GaussianSIFTMap)

            # shutil.copyfile(_nb_x_root, _output_nb_x_root)
            # shutil.copyfile(_nb_y_root, _output_nb_y_root)
            # shutil.copyfile(_b_x_root, _output_b_x_root)
            # shutil.copyfile(_b_y_root, _output_b_y_root)

        except Exception as e:
            try:
                dst_img2 = cv2.imread(dst_img_name)
                dst_rotate_img = rotate(dst_img2, - angle)
                img2 = cv2.cvtColor(dst_rotate_img, cv2.COLOR_BGR2GRAY)


                src_pts, dst_pts = SIFTwithRANSAC(img1, img2,
                                                  threshold=threshold,
                                                  MIN_MATCH_COUNT=MIN_MATCH_COUNT)
                # print(src_pts, dst_pts)
                rotate_GaussianSIFTMap = drawGaussianSIFTMap(src_pts, dst_pts, Gaussian=False, kernel=5)

                # cv2.imwrite(save_path + 'rotate_map.bmp', rotate_GaussianSIFTMap)
                _GaussianSIFTMap = rotate(rotate_GaussianSIFTMap, angle)

                cv2.imwrite(save_full_path, _GaussianSIFTMap)
                cv2.imwrite(img_new_path, _GaussianSIFTMap)

                # shutil.copyfile(_nb_x_root, _output_nb_x_root)
                # shutil.copyfile(_nb_y_root, _output_nb_y_root)
                # shutil.copyfile(_b_x_root, _output_b_x_root)
                # shutil.copyfile(_b_y_root, _output_b_y_root)

            except:
                print('FAIL File', dst_img_name)
                img = np.zeros((36, 36, 3), np.uint8)
                cv2.imwrite(save_full_path, img)

                pass


if __name__ == "__main__":
    inputs = list(range(0, num_process))
    pool = Pool(num_process)
    print("START!")
    start_time = time.time()

    pool_outputs = pool.map(job, inputs)

    print('full_path', len(os.listdir(output_full_folder_root)))
    print('move path', len(os.listdir(dst_new_folder_root)))
    print('_output_nb_x_root', len(os.listdir(output_nb_x_root)))
    print('_output_nb_y_root', len(os.listdir(output_nb_y_root)))
    print('_output_b_x_root', len(os.listdir(output_b_x_root)))
    print('_output_b_y_root', len(os.listdir(output_b_y_root)))


    end_time = time.time()
    print("Total Spend time：", str((end_time - start_time) / 60)[0:6] + "分鐘")

    print("Done")


    
