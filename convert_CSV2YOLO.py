#####################################################
#   Alex Denton, 16 Nov 2021, AE4824                #
#   this file coverts a CSV into YOLO format        #
#   heavily modified version of: https://github.com/karolmajek/YoloV3-Open-Images-v4/blob/master/convert-csv-to-yolo.py
#####################################################

import csv
import os  # for mkdir
from tqdm import tqdm


##### Define the given data structure #####
source_dir = "./data_given"
order = ['train', 'valid']

label_file = {
    'train': "./data_given/labels_train.csv",
    'valid': "./data_given/labels_val.csv",
}


##### Check for Directories and make if required #####
for mode in order:
    if not os.path.exists(f'{source_dir}/{mode}/'):
        os.mkdir(os.path.join(source_dir,mode))
        os.mkdir(os.path.join(source_dir,mode,'images'))  # top directory implies that images/ and labels/ also exist
        os.mkdir(os.path.join(source_dir,mode,'labels'))


##### Manually Defining Categories #####
classes_coded = [str(x) for x in list(range(1,12))]  # manually assign the numbers 1-11 as strings
classes_names = ['biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green', 'trafficLight-GreenLeft', 'trafficLight-Red', 'trafficLight-RedLeft', 'trafficLight-Yellow', 'trafficLight-YellowLeft', 'truck']
# classes_coded = []
#classes_names = []
#
# for l in csv.reader(open('./data_given/labels_train.csv')):
#     # print(l)
#     classes_coded.append(l[0])
#     classes_names.append(l[1])
#     # break
# print(len(classes_names))
print(classes_names)

####### Reframe Train and Test Data ######
for mode in order:
    input_file = csv.DictReader(open(label_file[mode]))

    for line in tqdm(list(input_file)):
        # print(line)
        # print(line['LabelName'],classes_coded.index(line['LabelName']))
        img_number = line['frame'].replace('.txt','')  # strips the '.txt' from the end of the file name

        # open/make a new txt file for the given image
        with open(f"{source_dir}/{mode}/labels/{img_number}.txt",'w') as labels:
            # format from https://medium.com/nerd-for-tech/day-86-dl-custom-object-detector-setup-yolov5-4f5539dd7e9a
            labels.write(' '.join([str(classes_coded.index(line['class_id'])), str(( float(line['xmax'])+float(line['xmin'])) /2 ), str(( float(line['ymax'])+float(line['ymin'])) /2 ),str(float(line['xmax'])-float(line['xmin'])),str(float(line['ymax'])-float(line['ymin']))])+'\n')

        # move the image (if not already done)
        if not os.path.exists(f"{source_dir}/{mode}/images/{img_number}.jpg"):
            os.rename(f"{source_dir}/{img_number}.jpg", f"{source_dir}/{mode}/images/{img_number}.jpg")  # CAUTION - this will delete the original
            #print(f'{line["frame"]} has been moved to {source_dir}/test/images/')

        #print(f'{source_dir}/test/labels/{line["frame"]}.txt has been written')
        # break


##### Create YAML File #####
with open(f'./{source_dir}/data.yaml','w') as yaml:
    yaml.write('\n'.join([
        'train: '+f'{source_dir}/train/images',
        'val: '+f'{source_dir}/valid/images',
        '\n',
        'nc: '+str(len(classes_coded)),
        'names: '+f'{classes_names}',
        ]))