import fingerprint_enhancer  # Load the library
import cv2
import os
import numpy as np
import threading
from multiprocessing import Process, Pool
import time

file_path = r"/data/FocalTech/testset_nasic9395_v7_merge/enroll/enroll_3/"

result_path = r"/data/FocalTech/testset_nasic9395_v7_merge/enroll_binary/enroll_3/"
num_process = 8
padding = 40

os.makedirs(result_path, exist_ok=True)
num_imgs = len(os.listdir(file_path))

files = os.listdir(file_path)

if '.DS_Store' in files:
    files.remove('.DS_Store')

for i in files:
    if i.split('.')[-1]!='bmp':
        files.remove(i)

def job(id):
    # file_path = files
    # imgs = os.listdir(file_path)[num_imgs * id // num_process: num_imgs * (id + 1) // num_process]
    imgs = files[num_imgs * id // num_process: num_imgs * (id + 1) // num_process]
    i=0
    for img_filename in imgs:


        img = cv2.imread(file_path + img_filename, 0)
        print(file_path + img_filename)
        img_padding = np.ones((img.shape[0] + padding * 2, img.shape[1] + padding * 2)) * 95
        img_padding[padding + 1: img.shape[0] + padding + 1, padding + 1: img.shape[1] + padding + 1] = img

        try:
            out = fingerprint_enhancer.enhance_Fingerprint(img_padding, resize=False)
        except:
            continue
        
        cv2.imwrite(result_path + img_filename,
                    out[padding + 1: img.shape[0] + padding + 1, padding + 1: img.shape[1] + padding + 1])
        i=i+1
        # print("i",i)

if __name__ == "__main__":
    inputs = [0, 1, 2, 3, 4, 5, 6, 7]
    pool = Pool(num_process)
    print("START!")
    start_time = time.time()

    pool_outputs = pool.map(job, inputs)
    end_time = time.time()
    print("Total Spend time：", str((end_time - start_time) / 60)[0:6] + "分鐘")

    print("Done")
