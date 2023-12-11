import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import argparse

def getBlack(img):
    height, width = img.shape
    black = 0
    for y in range(height):
        for x in range(width):
            pixel = img[y, x]
            if pixel < 10:
                black += 1
    percent = black / (height * width)
    percent = percent * 100

    return percent, black


def readData(dirName):
    files = os.listdir(dirName)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    black_pixel = []
    black_ratio = []
    name = []
    i = 0
    for path, currentDirectory, files in os.walk(dirName):
        # for folder in currentDirectory:
        for file in files:
            if file.endswith('.bmp'):
                image_root = os.path.join(path, file)
                img = cv2.imread(image_root, cv2.IMREAD_GRAYSCALE)
                percent, black = getBlack(img)
                black_pixel.append(black)
                black_ratio.append(percent)
                name.append(image_root)
                i = i + 1
                print(i)
                print(image_root)

    return black_pixel, black_ratio, name


def std_mean(black_ratio):
    std = np.std(black_ratio)
    mean = np.mean(black_ratio)

    return std, mean

def allFolder(a, b):
    array1 = np.hstack([a, b])

    return array1


def getSNR(dirName):
    files = os.listdir(dirName)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    snr = []
    for path, currentDirectory, files in os.walk(dirName):
        # for folder in currentDirectory:
        for file in files:
            if file.endswith('.bmp'):
                image_root = os.path.join(path, file)
                img = cv2.imread(image_root, cv2.IMREAD_GRAYSCALE)
                image = np.asanyarray(img)
                std, mean = std_mean(image)
                snr.append(mean / std)
                print(image_root)
    return snr


def main(args):
    ### mode = ['SNR', 'BLACK']
    # mode = 'BLACK'
    # all_dirName = '/Users/antinghsieh/Downloads/Testtool_align_crop/non_binary/train'
    dirName_1 = "/data/FocalTech/nasic9395_1008/noise"
    dirName_2 = "/data/FocalTech/nasic9395_1008/clear"
    file_type = dirName_1.split("/")[-1]
    save_name = dirName_1.split("/")[-2]
    route = dirName_1.replace(file_type, '')

    if args.mode == 'BLACK':
        pixel_1, ratio_1, name_1 = readData(dirName_1)
        std_1, mean_1 = std_mean(ratio_1)

        pixel_2, ratio_2, name_2 = readData(dirName_2)
        std_2, mean_2 = std_mean(ratio_2)

        all_ratio = allFolder(ratio_1, ratio_2)
        all_std, all_mean = std_mean(all_ratio)

    if args.mode == 'SNR':
        # all_ratio = getSNR(all_dirName)
        # all_std, all_mean = std_mean(all_ratio)

        ratio_1 = getSNR(dirName_1)
        std_1, mean_1 = std_mean(ratio_1)

        ratio_2 = getSNR(dirName_2)
        std_2, mean_2 = std_mean(ratio_2)

    # weights = np.ones_like(all_ratio) / len(all_ratio)
    plt.hist(all_ratio, alpha=0.5, density=True, bins=30, label='All', color='tab:orange')
    plt.hist(ratio_1, alpha=0.3, density=True, bins=30, label='Noised', color='tab:blue')
    plt.hist(ratio_2, alpha=0.3, density=True, bins=30, label='Normal', color='tab:green')

    domain = np.linspace(np.min(all_ratio), np.max(all_ratio))
    plt.plot(domain, norm.pdf(domain, all_mean, all_std),
             label='All ' + f'$( \mu \\approx{round(all_mean, 4)}, \sigma \\approx{round(all_std, 4)})$',
             color='tab:orange')
    #
    domain_1 = np.linspace(np.min(ratio_1), np.max(ratio_1))
    plt.plot(domain_1, norm.pdf(domain_1, mean_1, std_1),
             label='Noised ' + f'($ \mu \\approx{round(mean_1, 4)}, \sigma \\approx{round(std_1, 4)})$',
             color='tab:blue')

    domain_2 = np.linspace(np.min(ratio_2), np.max(ratio_2))
    plt.plot(domain_2, norm.pdf(domain_2, mean_2, std_2),
             label='Normal ' + f'$( \mu \\approx{round(mean_2, 4)}, \sigma \\approx{round(std_2, 4)})$',
             color='tab:green')
    # plt.plot(bins, 1 / (all_std * np.sqrt(2 * np.pi)) *
    #          np.exp(- (bins - all_mean) ** 2 / (2 * all_std ** 2)), color='y', label='line')
    plt.title(f"{save_name} Black Pixel Distribution")
    plt.xlabel("Ratio(%)")
    plt.ylabel("Amount")
    plt.legend()
    plt.savefig(f'{route}_{save_name}_{args.mode}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("End")


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='BLACK', choices=['SNR', 'BLACK'])
    args = parser.parse_args()

    main(args)

    end_time = time.time()
    print("Total Spend time：", str((end_time - start_time) / 60)[0:6] + "分鐘")
