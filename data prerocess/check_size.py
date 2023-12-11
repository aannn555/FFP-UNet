import cv2
import os

path = '/data/FocalTech/nasic9395_1018_v2_aug/non_binary/clean'

num_imgs = len(os.listdir(path))
files = os.listdir(path)

if '.DS_Store' in files:
    files.remove('.DS_Store')

i = 0
for img in os.listdir(path):

	input_img = cv2.imread(path + "/" + img, cv2.IMREAD_GRAYSCALE)

	# print(input_img.shape)

	if input_img.shape != (36,36):
		os.remove(path + "/" + img)
		i=i+1
		print(img)
print('i=',i)


