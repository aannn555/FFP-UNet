import shutil
import os

###移動檔案資料


destination = '/data/FocalTech/nasic9395_0606/original_data/merge_noise'
os.makedirs(destination, exist_ok=True)

counter = 0
for i in range(0, 10):
    source =f'/data/FocalTech/nasic9395_0606/original_data/0006_anting/{i}/noise'
    files = os.listdir(source)
    for file in files:
        new_path = shutil.copyfile(f"{source}/{file}", f"{destination}/{file}")
        # new_path = destination + file
        print(new_path)


destination_files = os.listdir(destination)
print('move numbers: ',len(destination_files))
print('Finish!!!')
