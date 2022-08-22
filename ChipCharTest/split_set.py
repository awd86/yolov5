##########################
# by/for Alex Denton, thesis work
# last modified 21 Aug 2022
##########################

"""
The newest revision of this file developes a full train/validate/test set based on ratios from: https://www.v7labs.com/blog/train-validation-test-set

Alternatively, I also found code that might replace this entire file here:
https://blog.paperspace.com/train-yolov5-custom-data/
"""

import os
import random
import shutil  # for the copy command
import yaml


def split_set(src_dir, split=[0.8, 0.1, 0.1], method='rand'):
    """
    Args:
        src_dir: Where to find the dataset to modify.
        split: [train,val,test] breakdown in percentages
        method: type of split to perform - basic (FIFO), rand (random), cross (cross-validation)

    Returns: Outputs in file structure
    """

    # Gather Image Files
    image_files = []
    for _file in os.listdir(path=f'{src_dir}/images/'):  # ["[image number].tif",...]
        if _file.endswith('.png'):
            image_files.append(_file[0:-4])  # remove suffix
            type = _file.split('.')[-1]  # but keep track of the suffix (assumes they're all the same...)

    # Gather Label Files
    label_files = []
    for _file in os.listdir(path=f'{src_dir}/labels/'):  # ["[image number].txt",...]
        if _file.endswith('.txt'):
            label_files.append(_file[0:-4])  # remove suffix

    #print(f'labels:\n{image_files[0:2]}')
    #print(f'labels:\n{label_files[0:2]}')

    # Create Six Directories
    levels = ['train', 'val', 'test']

    for m in levels:
        if not os.path.exists(f'{src_dir}/{m}'):
            os.mkdir(f'{src_dir}/{m}')
            
        if not os.path.exists(f'{src_dir}/{m}/images'):
            os.mkdir(f'{src_dir}/{m}/images')

        if not os.path.exists(f'{src_dir}/{m}/labels'):
            os.mkdir(f'{src_dir}/{m}/labels')


    # Handle 'split'
    if split[0] >= 1 and split[0] <101:  # train is percent, not fraction
        split = [x/100 for x in split]
    elif split[0] >=101:  # train is literal
        len_split = len(image_files)
        if not sum(split) == len_split:  # check for full sum
            split[2] = split[0] - split[1]  # if the numbers don't add up, take it out on the 'test' portion
        split = [x/len_split for x in split]


    # Move Files
    moves = [int(x*len(image_files)) for x in split]
    moves[2] = len(image_files)-moves[0]-moves[1]  # make sure no one is left out
    for m in range(3):
        level = levels[m]

        for image in range(moves[m]):

            # Determine which image to move based on 'method'
            if method == 'basic' or m == 2:  # all 'test' images are leftovers
                _image = 0  # will always take the first item in image_files as it is consumed
            elif method == 'rand':
                _image = random.randint(0, len(image_files)-1)  # image_files is getting smaller as they're consumed
            elif method == 'cross':
                print('Cross-Validation method has not been built. Reverting to random method.')
                _image = random.randint(0, len(image_files)-1)  # image_files is getting smaller as they're consumed
                method = 'rand'  # fix it for next time 'round the loop

            # Move Image File
            if image_files[_image] in label_files:  # only move files with a corresponding label
                shutil.move(f'{src_dir}/images/{image_files[_image]}.{type}',
                          f'{src_dir}/{level}/images/{image_files[_image]}.{type}')
                #print(f"moved image {_image}")
                # Move Label File
                shutil.move(f'{src_dir}/labels/{image_files[_image]}.txt',
                          f'{src_dir}/{level}/labels/{image_files[_image]}.txt')
                # Remove from set
                image_files.pop(_image)

            else:
                print(f"{image_files[_image]} not found in labels")
                image_files.pop(_image)  # remove from set to prevent recurring error

    # update data.yaml with new directories
    update_yaml(src_dir)
    
    # remove empty folders
    try:
        os.rmdir(f'{src_dir}/labels')
        os.rmdir(f'{src_dir}/images')
    except:
        print(f'Unable to delete empty folders in {src_dir}')
                
    print(f'The {src_dir} move is complete')
    
    
    
# Overwrite the data.yaml file
def update_yaml(src_dir):
    
    # Find list of all classes to put in data.yaml (this will ONLY work if the yaml was saved during image creation)
    with open(f'{src_dir}/data.yaml', 'r') as old_yaml:
        data  = yaml.full_load(old_yaml)
        nc = data.get('nc')
        names = data.get('names')

    # Create/Overwrite YAML File
    with open(f'{src_dir}/data.yaml', 'w') as new_yaml:
        new_yaml.write('\n'.join([

            'train: ' + f'{src_dir}/train/images',
            'val: ' + f'{src_dir}/val/images',
            'test: ' + f'{src_dir}/test/images',

            '\n',
            'nc: ' + str(nc),
            'names: ' + str(names),
            ]))
        
        
    

##### Execution #####
if __name__ == '__main__':
    src_dir = 'Set1_M35_2s1_M1/chips_05_black'
    split_set(src_dir)
    #update_yaml(src_dir)

