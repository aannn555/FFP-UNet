import os
import numpy as np
import sys

samp_path = "C:\\python_progress\\fingerprint_recognition\\model_v10\\0707ft_lightnoised_chao_identify\\0707ft_lightnoised\\0709_enhanced\\pgt_block_48\\ft_lightnoised_157x36_pretrain\\total_loss_mse_ssim\\non_binary\\samp"

def main():
    cnt_array = np.zeros((2, 10), np.int)

    for img_filename in os.listdir(samp_path):
        filename_split = img_filename.split("_")
        person_id = filename_split[2]
        finger_id = int(filename_split[3])

        print("person_id = ", person_id)
        print("finger_id = ", finger_id)


        if(person_id == "0001"):
            cnt_array[0][finger_id] = cnt_array[0][finger_id] + 1
        elif(person_id == "0002"):
            cnt_array[1][finger_id] = cnt_array[1][finger_id] + 1
        else:
            print("error!")
            sys.exit()
    
    for i in range(2):
        for j in range(10):
            print("person_id = ", i + 1, ", finger_id = ", j, ", success = ", cnt_array[i][j])

if __name__ == "__main__":
    main()