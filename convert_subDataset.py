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
from tqdm import tqdm
# import yaml
# import sys
# import ruamel.yaml


# desired Training Dataset Size
set_size = 22000

##### Define the given data structure #####
source_dir = "./data_car_small/"
source_img_dir = f"{source_dir}/export/images/"
source_txt_dir = f"{source_dir}/export/labels/"
dest_dir = "./data_car_small_sub/"
#order = ['train', 'test', 'valid']
order = ['train', 'valid']


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
all_images = [x.replace('.jpg','') for x in all_images]  # remove '.jpg' from the titles
#print(all_images)


####### Reframe Train and Test Data ######
#real_set_size = 1.4*set_size  # 100% set +20% test +20% valid
real_set_size = int(1.2*set_size)
if real_set_size > len(all_images):
    #print(f"Requested set size of {set_size} doesn't allow 20% test and 20% validation. Only {len(all_images)} available.")
    #print(f"Resetting to maximum set size of {len(all_images)*0.6}")
    #set_size = len(all_images)*0.6
    print(f"Requested set size of {set_size} doesn't allow 20% validation. Only {len(all_images)} available.")
    print(f"Resetting to maximum set size of {len(all_images)*0.8}")
    set_size = len(all_images)*0.8
    real_set_size = len(all_images)

# randomize the sample set
shuf_set = list(range(real_set_size))
random.shuffle(shuf_set)


##### Copy/Paste Image and Label Files #####
for i in tqdm(range(real_set_size)):  # makes test and valid at 20% size each
    n = shuf_set[i]  # random sample

    if i <= set_size:
        mode = order[0]  # 'train'
    elif i <= 1.2*set_size:
        mode = order[1]  # 'test' (or valid)
    else:
        mode = order[2]  # 'valid'

    shutil.copyfile(f"{source_img_dir}/{all_images[n]}.jpg", f"{dest_dir}/{mode}/images/{all_images[n]}.jpg")  # image
    # print(f'{all_images[n]} has been moved')

    for label_file in glob.glob(f'{source_txt_dir}/{all_images[n]}.*'):
        # if i%1000 == 0: print(f'{label_file} has been moved')  # prints status every 1000 files
        shutil.copy(label_file, f"{dest_dir}/{mode}/labels/")


##### Make the YAML file #####
#shutil.copy(f'{source_dir}/data.yaml',f'{dest_dir}/data.yaml')

print('Edit your YAML file manually - this doc is not complete.')

# Edit the YAML File
# yaml = ruamel.yaml.YAML()
# # yaml.preserve_quotes = True
# with open(f'./{dest_dir}/data.yaml','w') as old:
#     data = yaml.load(old)
#     data['train'] = f'{dest_dir}/train/images'
#     data['val']  = f'{dest_dir}/valid/images'
# yaml.dump(data, sys.stdout)
#
# with open(f'./{dest_dir}/data.yaml') as old_yaml:
#     list_doc = yaml.safe_load(old_yaml)
#
# for data in list_doc:
#     data['train'] = f'{dest_dir}/train/images'
#     data['val'] = f'{dest_dir}/valid/images'
#
# with open(f'./{dest_dir}/data.yaml', 'w') as new_yaml:
#     yaml.dump(list_doc, new_yaml)