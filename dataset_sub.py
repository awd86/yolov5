#####################################################
#   Alex Denton, 16 Nov 2021, AE4824                #
#   this file coverts a CSV into YOLO format        #
#   heavily modified version of: https://github.com/karolmajek/YoloV3-Open-Images-v4/blob/master/convert-csv-to-yolo.py
#####################################################

import csv
import os  # for mkdir
from tqdm import tqdm

# desired Training Dataset Size
set_size = 5000

##### Define the given data structure #####
source_dir = "./data_given"
order = ['train', 'test', 'valid']

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

real_set_size = 1.4*set_size



####### Reframe Train and Test Data ######
for i in real_set_size:  # makes test and valid at 20% size each

    if i <= set_size:
        mode = order[0]  # 'train'
    elif i <= 1.2*set_size:
        mode = order[1]  # 'test'
    else:
        mode = order[2]  # 'valid'

    shutil.copyfile()


    input_file = csv.DictReader(open(label_file[mode]))

    for line in tqdm(list(input_file)):
        # print(line)
        # print(line['LabelName'],classes_coded.index(line['LabelName']))

        # open/make a new txt file for the given image
        with open(f"./data_given/{mode}/labels/{line['frame']}.txt",'w') as labels:
            # format from https://medium.com/nerd-for-tech/day-86-dl-custom-object-detector-setup-yolov5-4f5539dd7e9a
            labels.write(' '.join([str(classes_coded.index(line['class_id'])), str(( float(line['xmax'])+float(line['xmin'])) /2 ), str(( float(line['ymax'])+float(line['ymin'])) /2 ),str(float(line['xmax'])-float(line['xmin'])),str(float(line['ymax'])-float(line['ymin']))])+'\n')

        # move the image (if not already done)
        if not os.path.exists(f"./data_given/{mode}/images/{line['frame']}"):
            os.rename(f"./data_given/{line['frame']}", f"./data_given/{mode}/images/{line['frame']}")  # CAUTION - this will delete the original
            #print(f'{line["frame"]} has been moved to {source_dir}/test/images/')

        #print(f'{source_dir}/test/labels/{line["frame"]}.txt has been written')
        # break
