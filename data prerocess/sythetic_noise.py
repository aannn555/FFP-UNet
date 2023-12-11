import numpy as np
import cv2
import os
from PIL import ImageFilter, Image
import random
import skimage


############################## Parameter ####################################

#   MIN_KERNEL_SIZE / MAX_KERNEL_SIZE: The size of the process region       #
#   NOISE_DARKNESS: The color of the noise (0~1)                            #
#   NOISE_SOFT: The smoothness of the region edge (0~1)                     #
#   NOISE_PROBABILITY: The appearance probability of the noise (0~1)        #

#############################################################################


class Synthetic_Noise():
    def create_gaussian_kernel(self):
        for i in range(self.MIN_KERNEL_SIZE, self.MAX_KERNEL_SIZE + 1, 2):
            kernel_size = i
            sig = self.SIGMA + i * 0.25
            r1, r2 = -(kernel_size - 1) // 2, -(kernel_size - 1) // 2 + kernel_size
            x, y = np.mgrid[r1:r2, r1:r2]
            self.gaussian_kernels.append(np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(sig)))


class NS(Synthetic_Noise):
    def __init__(self):
        self.MAX_KERNEL_SIZE = 27
        self.MIN_KERNEL_SIZE = 7
        self.SIGMA = 2.
        self.NOISE_DARKNESS = 0
        self.DARKNESS_RANGE = 0.1
        self.NOISE_SOFT = 0.1
        self.NOISE_PROBABILITY = 0.04
        self.BOARD = self.MAX_KERNEL_SIZE // 2
        self.gaussian_kernels = []
        self.create_gaussian_kernel()

    def draw_img(self, img: np.ndarray, mask: np.ndarray):
        img_size = img.shape
        mask = np.where(mask == 0, self.NOISE_SOFT, mask)
        for px in range(0 + self.BOARD, img_size[0] - self.BOARD):
            for py in range(0 + self.BOARD, img_size[1] - self.BOARD):
                kernel = self.gaussian_kernels[np.random.randint(0, len(self.gaussian_kernels))]
                kernel_size = kernel.shape[0:2]

                if mask[px, py] == 1 and np.random.uniform(low=0, high=9, size=1)[0] < self.NOISE_PROBABILITY:
                    sharp = self.NOISE_DARKNESS + \
                            np.random.uniform(low=-self.DARKNESS_RANGE, high=self.DARKNESS_RANGE, size=1)[0]
                    rx1, rx2 = -(kernel_size[0] - 1) // 2, -(kernel_size[0] - 1) // 2 + kernel_size[0]
                    ry1, ry2 = -(kernel_size[1] - 1) // 2, -(kernel_size[1] - 1) // 2 + kernel_size[1]
                    img[px + rx1: px + rx2, py + ry1: py + ry2] -= np.multiply(
                        np.multiply(img[px + rx1: px + rx2, py + ry1: py + ry2], kernel * sharp),
                        mask[px + rx1: px + rx2, py + ry1: py + ry2])
        return img


class NS_v1(Synthetic_Noise):
    def __init__(self):
        self.MAX_KERNEL_SIZE = 21
        self.MIN_KERNEL_SIZE = 13
        self.SIGMA = 1.
        self.NOISE_DARKNESS = -0.2
        self.DARKNESS_RANGE = 0.01
        self.NOISE_SOFT = 0.55
        self.NOISE_PROBABILITY = 0.2
        self.BOARD = self.MAX_KERNEL_SIZE // 2
        self.gaussian_kernels = []
        self.create_gaussian_kernel()

    def draw_img(self, img: np.ndarray, mask: np.ndarray):
        img_size = img.shape
        mask = np.where(mask == 0, self.NOISE_SOFT, mask)
        for px in range(0 + self.BOARD, img_size[0] - self.BOARD):
            for py in range(0 + self.BOARD, img_size[1] - self.BOARD):
                kernel = self.gaussian_kernels[np.random.randint(0, len(self.gaussian_kernels))]

                kernel_size = kernel.shape[0:2]

                if np.random.uniform(low=0, high=9, size=1)[0] < self.NOISE_PROBABILITY:
                    if mask[px, py] == 1:
                        sharp = self.NOISE_DARKNESS + \
                                np.random.uniform(low=-self.DARKNESS_RANGE, high=self.DARKNESS_RANGE, size=1)[0]
                        rx1, rx2 = -(kernel_size[0] - 1) // 2, -(kernel_size[0] - 1) // 2 + kernel_size[0]
                        ry1, ry2 = -(kernel_size[1] - 1) // 2, -(kernel_size[1] - 1) // 2 + kernel_size[1]
                        img[px + rx1: px + rx2, py + ry1: py + ry2] += np.multiply(kernel * sharp,
                                                                                   mask[px + rx1: px + rx2,
                                                                                   py + ry1: py + ry2])

        return img


class NS_v2(NS_v1):
    def draw_img(self, img: np.ndarray, mask: np.ndarray):
        img_size = img.shape
        mask = np.where(mask == 0, self.NOISE_SOFT, mask)
        for px in range(0 + self.BOARD, img_size[0] - self.BOARD):
            for py in range(0 + self.BOARD, img_size[1] - self.BOARD):
                kernel = self.gaussian_kernels[np.random.randint(0, len(self.gaussian_kernels))]
                kernel_size = kernel.shape[0:2]

                if mask[px, py] == 1 and np.random.uniform(low=0, high=9, size=1)[0] < self.NOISE_PROBABILITY:
                    sharp = self.NOISE_DARKNESS + \
                            np.random.uniform(low=-self.DARKNESS_RANGE, high=self.DARKNESS_RANGE, size=1)[0]
                    rx1, rx2 = -(kernel_size[0] - 1) // 2, -(kernel_size[0] - 1) // 2 + kernel_size[0]
                    ry1, ry2 = -(kernel_size[1] - 1) // 2, -(kernel_size[1] - 1) // 2 + kernel_size[1]
                    img[px + rx1: px + rx2, py + ry1: py + ry2] += np.multiply(kernel * sharp, mask[px + rx1: px + rx2,
                                                                                               py + ry1: py + ry2])

                    # Create gaussian blur region
                    mask_blur = img.copy() * 0
                    cv2.circle(mask_blur, (px, py), 7, (1), -1)
                    blur = img.copy()
                    blur = cv2.GaussianBlur(blur, (9, 9), 2.3) * 1.05
                    # blur = cv2.bilateralFilter(blur, 9, 2.3, 2.3)* 1.05
                    img = np.where(mask_blur, blur, img)
        return img


class NS_v3(Synthetic_Noise):
    def __init__(self):
        self.MAX_KERNEL_SIZE = 21
        self.MIN_KERNEL_SIZE = 13
        self.SIGMA = 1.
        self.NOISE_DARKNESS = 0.08
        self.DARKNESS_RANGE = 0.01
        self.NOISE_SOFT = 0.55
        self.NOISE_PROBABILITY = 0.05
        self.BOARD = self.MAX_KERNEL_SIZE // 2
        self.gaussian_kernels = []
        self.create_gaussian_kernel()

    def draw_img(self, img: np.ndarray, mask: np.ndarray):
        img_size = img.shape
        mask = np.where(mask == 0, self.NOISE_SOFT, mask)
        for px in range(0 + self.BOARD, img_size[0] - self.BOARD):
            for py in range(0 + self.BOARD, img_size[1] - self.BOARD):
                kernel = self.gaussian_kernels[np.random.randint(0, len(self.gaussian_kernels))]
                kernel_size = kernel.shape[0:2]

                if mask[px, py] == 1 and np.random.uniform(low=0, high=9, size=1)[0] < self.NOISE_PROBABILITY:
                    sharp = self.NOISE_DARKNESS + \
                            np.random.uniform(low=-self.DARKNESS_RANGE, high=self.DARKNESS_RANGE, size=1)[0]
                    rx1, rx2 = -(kernel_size[0] - 1) // 2, -(kernel_size[0] - 1) // 2 + kernel_size[0]
                    ry1, ry2 = -(kernel_size[1] - 1) // 2, -(kernel_size[1] - 1) // 2 + kernel_size[1]
                    img[px + rx1: px + rx2, py + ry1: py + ry2] += np.multiply(kernel * sharp, mask[px + rx1: px + rx2,
                                                                                               py + ry1: py + ry2])

                    # Create gaussian blur region
                    mask_for_blur = np.zeros(img.shape)
                    cv2.circle(mask_for_blur, (px, py), 7, 1, -1)
                    blur_img = cv2.GaussianBlur(img, (9, 9), 2.3) * 1.05
                    # blur_img = cv2.bilateralFilter(img, 9, 2.3, 2.3) * 1.05
                    img = np.where(mask_for_blur, blur_img, img)

                    # Add white noise on gaussian blur region
                    noise_img = skimage.util.random_noise(image=img, mode='speckle', clip=True, mean=0, var=0.01)
                    img = np.where(mask_for_blur, noise_img, img)
        return img


if __name__ == '__main__':
    synthetic_creator = NS_v1()

    size = 16
    type = 'enroll'
    normal_path = f"./0518synthetic_data/36/all_angles"
    mask_path = f"./0518synthetic_data/36/binary_mask"
    output_path = f"./0518synthetic_data/36/noised"
    # output_path = f"./0512synthetic_data/noised/{type}/{size}"

    files = os.listdir(normal_path)

    if '.DS_Store' in files:
        files.remove('.DS_Store')

    for img_file in files:
        img = cv2.imread(normal_path + "/" + img_file, cv2.IMREAD_GRAYSCALE)
        masked = cv2.imread(mask_path + "/" + img_file, cv2.IMREAD_GRAYSCALE)

        img = img / 255.
        masked = masked / 255

        noise_img = synthetic_creator.draw_img(img, masked)
        os.makedirs(os.path.join(output_path), exist_ok=True)
        cv2.imwrite(output_path + "/" + img_file, noise_img * 255)
