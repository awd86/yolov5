import os


def split_set(src_dir,testing):
    
    # Gather Image Files
    image_files = []
    for _file in os.listdir(path=f'{src_dir}/original_images/train'):  # ["[image number].tif",...]
        if _file.endswith('.tif' or '.jpg'):
            image_files.append(_file.split('.')[0])  # no suffix
            type = _file.split('.')[-1]  # keep track of the suffix (hopefully they're all the same...)

    # Gather Label Files
    label_files = []
    for _file in os.listdir(path=f'{src_dir}/original_labels/train'):  # ["[image number].tif",...]
        if _file.endswith('.txt'):
            label_files.append(_file.split('.')[0])  # no suffix
            
    # Create Test Directories
    if not os.path.exists(f'{src_dir}/original_images/test'):
        os.mkdir(f'{src_dir}/original_images/test')
    img_dir = f'{src_dir}/original_images/test'
    
    if not os.path.exists(f'{src_dir}/original_labels/test'):
        os.mkdir(f'{src_dir}/original_labels/test')
    lbl_dir = f'{src_dir}/original_labels/test'
            
    # Handle 'testing'
    if testing >= 1:  # train is literal, not percentage
        testing = testing/len(label_files)  # convert to percentage
            
    # Move Files
    for tester in range(int(len(image_files)*testing)):  # only move the percentage specified in 'train'
        if image_files[tester] in label_files:  # only move files with a corresponding label
            # Move Image File
            os.rename(f'{src_dir}/original_images/train/{image_files[tester]}.{type}', 
                      f'{img_dir}/{image_files[tester]}.{type}')
            # Move Label File
            os.rename(f'{src_dir}/original_labels/train/{image_files[tester]}.txt',
                      f'{lbl_dir}/{image_files[tester]}.txt')
        else:
            print(f"{image_files[tester]}not found in labels")

    print('The move is complete')
    

##### Execution #####
# split_set('data_xView',0.2)