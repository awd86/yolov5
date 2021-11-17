#####################################################
#   Alex Denton, 16 Nov 2021, AE4824                #
#   this file coverts a CSV into YOLO format        #
#   heavily modified version of: https://github.com/karolmajek/YoloV3-Open-Images-v4/blob/master/convert-csv-to-yolo.py
#####################################################

import csv
import os  # for mkdir
import random
import shutil
import glob


# desired Training Dataset Size
set_size = 5000

##### Define the given data structure #####
source_dir = "./data_given/"
source_img_dir = "./data_given/test/images/"
source_txt_dir = "./data_given/test/labels/"
dest_dir = "./data_car_sub/"
order = ['train', 'test', 'valid']


##### Check for Directories and make if required #####
if not os.path.exists(f'{dest_dir}/'):
    os.mkdir(os.path.join(dest_dir))

for mode in order:
    if not os.path.exists(f'{dest_dir}/{mode}/'):
        os.mkdir(os.path.join(dest_dir,mode))
        os.mkdir(os.path.join(dest_dir,mode,'images'))  # top directory implies that images/ and labels/ also exist
        os.mkdir(os.path.join(dest_dir,mode,'labels'))


##### Make a list of all image titles in the folder #####
all_images = os.listdir(path=f'{source_img_dir}')  # ["[image number].jpg",...]
#print(all_images)


####### Reframe Train and Test Data ######
real_set_size = 1.4*set_size
if real_set_size > len(all_images):
    print(f"Requested set size of {set_size} doesn't allow 20% test and 20% validation. Only {len(all_images)} available.")
    print(f"Resetting to maximum set size of {len(all_images)*0.6}")
    set_size = len(all_images)*0.6
    real_set_size = len(all_images)

# randomize the sample set
shuf_set = list(range(real_set_size))
random.shuffle(shuf_set)


##### Copy/Paste Image and Label Files #####
for i in range(real_set_size):  # makes test and valid at 20% size each
    n = shuf_set[i]  # random sample

    if i <= set_size:
        mode = order[0]  # 'train'
    elif i <= 1.2*set_size:
        mode = order[1]  # 'test'
    else:
        mode = order[2]  # 'valid'

    shutil.copyfile(f"{source_img_dir}/{all_images[n]}", f"{dest_dir}/{mode}/images/{all_images[n]}")  # image
    # print(f'{all_images[n]} has been moved')

    for label_file in glob.glob(f'{source_txt_dir}/{all_images[n]}.*'):
        #print(f'{label_file} has been moved')
        shutil.copy(label_file, f"{dest_dir}/{mode}/labels/")


##### Move the YAML file #####
shutil.copy(f'{source_dir}/data.yaml',f'{dest_dir}/data.yaml')