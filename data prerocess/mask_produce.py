import cv2
import os

size = 16
normal_path = f"./nasic9395_0613_del_heavy_noise_aug_v3/y_val"
mask_path = f"./nasic9395_0613_del_heavy_noise_aug_v3/y_val_mask"

os.makedirs(mask_path, exist_ok=True)

files = os.listdir(normal_path)
if '.DS_Store' in files:
    files.remove('.DS_Store')

count = 0
for img_file in files:

    img = cv2.imread(normal_path + "/" + img_file, cv2.IMREAD_GRAYSCALE)
    print('count = ', count, img_file, img.shape)
    # print(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 135:
                img[i][j] = 0
            else:
                img[i][j] = 255
    os.makedirs(os.path.join(mask_path), exist_ok=True)
    cv2.imwrite(mask_path + "/" + img_file, img)
    count = count+1
print('Finish!!!')
