#####################################################
#   Alex Denton, 16 Nov 2021, AE4824                #
#   this file coverts a JSON into YOLO format       #
#####################################################

import json
import os  # for mkdir
from tqdm import tqdm

# https://www.geeksforgeeks.org/read-json-file-using-python/
file = open('data_xView/xView_train.geojson')
labels = json.load(file)

features = labels['features'][2]
print(features)

file.close()