#####################################################
#   Alex Denton, 3 Dec 2021, AE4824                 #
#   re_classify uses a lookup table to change       #
#   classifications in label.txt files              #
#####################################################

import numpy as np
import os
import math
import pandas as pd
import warnings
from pathlib import Path
from vague_classes import convert_strip, convert_gl, convert_hhg  # this is the conversion dict


def goldilocks(lbl_dir,dest_dir,convert_dict):
    # 'lbl_dir' is the path to the folder containing the labels
    # 'convert' is a dict of old=new classification values BY NUMBER
    # to remove a class, the key value must be 'None'

    # make goldilocks directory
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)  # make a new directory parallel to lbl_dir

    # collect label file names
    label_files = []
    for file in os.listdir(path=f'{lbl_dir}'):  # ["[image number].tif",...]
        if file.endswith('.txt'):
            label_files.append(f'{lbl_dir}/{file}')

    # modify the label files
    for _label in label_files:

        # load labels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                labels = np.genfromtxt(_label, delimiter=' ')
            except:
                labels = []
            labels = np.atleast_2d(labels)
            #print(labels)

        # swap values
        Labels = []
        _labels = labels.copy()
        if labels.any():  # ignore blank files

            # key:value replacement
            _labels[:,0] = [convert_dict[cls] for cls in labels[:,0]]
            # print(_labels)

            # remove None class lines
            for k in range(len(_labels[:,0])):
                # print(f'k is {k}')
                if not math.isnan(_labels[k,0]):  # if not None
                    # print(f'k is {k}, class is {_labels[k,0]}')
                    if Labels == []:  # if no data yet
                        Labels = _labels[k,:]  # first row
                    else:
                        Labels = np.vstack([Labels, _labels[k,:]])

        # Convert the label array to a Pandas dataframe
        if not Labels == []:
            # print(Labels)
            Labels_pd = pd.DataFrame(np.atleast_2d(Labels),columns=['new class','xctr','yctr','xw','yh'])
            Labels_pd['new class'] = Labels_pd['new class'].map(lambda x: '%1.d' % x)  # no decimals in the 'new class' column

            # print(f"Pandas frame:\n {Labels_pd}")
            # print(Labels_pd)

        else:  # If no labels, make the file blank (YOLO format)
            Labels = np.array([])
            Labels_pd = pd.DataFrame(np.atleast_2d(Labels))

        # Write the new label file
        Labels_pd.to_csv(f"{dest_dir}/{_label.split('/')[-1]}", sep=' ',header=None, index=False, float_format='%.6f')

    print(f'\n{len(label_files)} files converted to goldilocks values.')


##### Testing #####
# goldilocks('data_testing/labels','data_testing/labels_01_stripped',convert_strip)
# goldilocks('data_testing/labels','data_testing/labels_02_goldilocks',convert_gl)
# goldilocks('data_testing/labels','data_testing/labels_03_hhg',convert_hhg)

##### Execution #####
# goldilocks('data_xView/labels/train','data_xView/labels/train_01_stripped',convert_strip)
# goldilocks('data_xView/labels/val','data_xView/labels/val_01_stripped',convert_strip)
# goldilocks('data_xView/labels/train','data_xView/labels/train_02_gl',convert_gl)
# goldilocks('data_xView/labels/val','data_xView/labels/val_02_gl',convert_gl)
# goldilocks('data_xView/labels/train','data_xView/labels/train_03_hhg',convert_hhg)
# goldilocks('data_xView/labels/val','data_xView/labels/val_03_hhg',convert_hhg)