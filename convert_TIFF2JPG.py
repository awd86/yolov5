#####################################################
#   Alex Denton, 26 Nov 2021, AE4824                #
#   this file coverts TIFF images to JPG format     #
#####################################################

import csv
import os  # for mkdir
from tqdm import tqdm
from PIL import Image

source_dir = 'H:/Dataset xView/val_images'
dest_dir = 'H:/Dataset xView/val_jpg_images'

##### Check for Directories and make if required #####
if not os.path.exists(f'{dest_dir}/'):
    os.mkdir(os.path.join(dest_dir))

##### Make a list of all image titles in the folder #####
all_images = os.listdir(path=f'{source_dir}')  # ["[image number].tif",...]
all_images = [img for img in all_images if '.tif' in img]  # removes all list elements that don't contain '.tif'
all_images = [x.replace('.tif', '') for x in all_images]  # remove '.tif' from the titles
#print(all_images)

##### Convert from TIFF to JPG #####
for img_name in tqdm(all_images):  # tqdm provides the status bar
    img = Image.open(f'{source_dir}/{img_name}.tif')
    img.save(f'{dest_dir}/{img_name}.jpg')

