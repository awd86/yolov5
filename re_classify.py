#####################################################
#   Alex Denton, 3 Dec 2021, AE4824                 #
#   re_classify uses a lookup table to change       #
#   classifications in label.txt files              #
#####################################################

import numpy as np
import os
from pathlib import Path
from vague_classes import convert  # this is the conversion dict


def goldilocks(lbl_dir,convert):
    # 'lbl_dir' is the path to the folder containing the labels
    # 'convert' is a dict of old=new categoris values BY NUMBER

    # make goldilocks directory
    src_dir = Path(lbl_dir).parent.absolute()
    if not os.path.exists(f'{src_dir}/goldilocks_labels'):
        os.mkdir(f'{src_dir}/goldilocks_labels')
    dest_dir = f'{src_dir}/goldilocks_labels'

    # collect label file names
    label_files = []
    for file in os.listdir(path=f'{lbl_dir}'):  # ["[image number].tif",...]
        if file.endswith('.txt'):
            label_files.append(f'{lbl_dir}/{file}')

    for _label in label_files:
        # load labels
        try:
            labels = np.genfromtxt(_label, delimiter=' ')
        except:
            labels = []
        labels = np.atleast_2d(labels)
        #print(labels)

        # swap values
        if labels.any():
            labels[:,0] = [convert[cls] for cls in labels[:,0]]  # key:value lookup

        # save modified labels
        np.savetxt(f"{dest_dir}/{_label.split('/')[-1]}", np.atleast_2d(labels), delimiter=' ', newline='\n',
                   encoding=None)

    print(f'\n{len(label_files)} files converted to goldilocks values.')

##### Testing #####
goldilocks('data_xView/labels',convert)


# TODO finish building HolyHandGrenade
def holyHandGrendade(lbl_dir,convert):
    # 'lbl_dir' is the path to the folder containing the labels
    # 'convert' is a dict of old=new categoris values BY NUMBER

    ##### in holyHandGrenade only values of '1' will remain. All other labels are DELETED #####

    # make goldilocks directory
    src_dir = Path(lbl_dir).parent.absolute()
    if not os.path.exists(f'{src_dir}/hhg_labels'):
        os.mkdir(f'{src_dir}/hhg_labels')
    dest_dir = f'{src_dir}/hhg_labels'

    # collect label file names
    label_files = []
    for file in os.listdir(path=f'{lbl_dir}'):  # ["[image number].tif",...]
        if file.endswith('.txt'):
            label_files.append(f'{lbl_dir}/{file}')

    for _label in label_files:
        # load labels
        try:
            labels = np.genfromtxt(_label, delimiter=' ')
        except:
            labels = []
        labels = np.atleast_2d(labels)
        #print(labels)

        # swap values
        if labels.any():
            labels[:,0] = [convert[cls] for cls in labels[:,0]]  # key:value lookup

        # save modified labels
        np.savetxt(f"{dest_dir}/{_label.split('/')[-1]}", np.atleast_2d(labels), delimiter=' ', newline='\n',
                   encoding=None)

    print(f'\n{len(label_files)} files converted to holyHandGrenade values.')
