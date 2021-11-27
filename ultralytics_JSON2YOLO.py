#####################################################
#   Alex Denton, 16 Nov 2021, AE4824                #
#   this file coverts a JSON into YOLO format       #
#   Derived from the bottom of the 'xView.yaml' file from https://github.com/ultralytics/yolov5/blob/master/data/xView.yaml
#####################################################


import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.datasets import autosplit
from utils.general import download, xyxy2xywhn


def convert_labels(fname=Path('data_xView/xView_train.geojson')):
    # Convert xView geoJSON labels to YOLO format
    path = fname.parent
    # print(f'path is {path}')
    with open(fname) as f:
        print(f'Loading {fname}...')
        data = json.load(f)

    # Make dirs
    labels = Path(path / 'labels' / 'train')
    os.system(f'rm -rf {labels}')  # remove existing directory, if it already exists
    labels.mkdir(parents=True, exist_ok=True)

    # xView classes 11-94 to 0-59
    xview_class2index = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10, 11,
                         12, 13, 14, 15, -1, -1, 16, 17, 18, 19, 20, 21, 22, -1, 23, 24, 25, -1, 26, 27, -1, 28, -1,
                         29, 30, 31, 32, 33, 34, 35, 36, 37, -1, 38, 39, 40, 41, 42, 43, 44, 45, -1, -1, -1, -1, 46,
                         47, 48, 49, -1, 50, 51, -1, 52, -1, -1, -1, 53, 54, -1, 55, -1, -1, 56, -1, 57, -1, 58, 59]

    shapes = {}
    for feature in tqdm(data['features'], desc=f'Converting {fname}'):
        p = feature['properties']
        if p['bounds_imcoords']:
            id = p['image_id']
            file = path / 'train_images' / id
            # print(f'the file is {file}')
            if file.exists():  # 1395.tif missing
                # print(f'{id} exists')
                try:
                    box = np.array([int(num) for num in p['bounds_imcoords'].split(",")])
                    assert box.shape[0] == 4, f'incorrect box shape {box.shape[0]}'
                    cls = p['type_id']
                    cls = xview_class2index[int(cls)]  # xView class to 0-60
                    assert 59 >= cls >= 0, f'incorrect class index {cls}'

                    # Write YOLO label
                    if id not in shapes:
                        shapes[id] = Image.open(file).size
                    box = xyxy2xywhn(box[None].astype(np.float), w=shapes[id][0], h=shapes[id][1], clip=True)
                    with open((labels / id).with_suffix('.txt'), 'a') as f:
                        f.write(f"{cls} {' '.join(f'{x:.6f}' for x in box[0])}\n")  # write label.txt
                    #     print('box appended')
                    # print('file printed')
                except Exception as e:
                    print(f'WARNING: skipping one label for {file}: {e}')
            # else:
            #     print(f'nothing found to print')
            #     break

# Download manually from https://challenge.xviewdataset.org
#dir = Path(yaml['path'])  # dataset root dir
dir = Path('H:/Dataset xView/')  # dataset root dir on USB drive
# urls = ['https://d307kc0mrhucc3.cloudfront.net/train_labels.zip',  # train labels
#         'https://d307kc0mrhucc3.cloudfront.net/train_images.zip',  # 15G, 847 train images
#         'https://d307kc0mrhucc3.cloudfront.net/val_images.zip']  # 5G, 282 val images (no labels)
# download(urls, dir=dir, delete=False)

# Convert labels
convert_labels(dir / 'xView_train.geojson')

# Move images
images = Path(dir / 'images')
images.mkdir(parents=True, exist_ok=True)
Path(dir / 'train_images').rename(dir / 'images' / 'train')
Path(dir / 'val_images').rename(dir / 'images' / 'val')

# Split
autosplit(dir / 'images' / 'train')
