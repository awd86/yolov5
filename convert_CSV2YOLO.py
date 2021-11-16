import csv
from tqdm import tqdm

# classes_coded = []
#classes_names = []
#
# for l in csv.reader(open('./data_given/labels_train.csv')):
#     # print(l)
#     classes_coded.append(l[0])
#     classes_names.append(l[1])
#     # break
# print(len(classes_names))

classes_coded = [str(x) for x in list(range(1,12))]
classes_names = ['biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green', 'trafficLight-GreenLeft', 'trafficLight-Red', 'trafficLight-RedLeft', 'trafficLight-Yellow', 'trafficLight-YellowLeft', 'truck']

print(classes_coded)

# 601 classes, not 600...

input_file = csv.DictReader(open("./data_given/labels_train.csv"))

for line in tqdm(list(input_file)):
    # print(line)
    # print(line['class_id'],classes_coded.index(line['class_id']))
    with open('./data_given/train/labels/%s.txt'%line['frame'],'w') as f:
        f.write(' '.join([str(classes_coded.index(line['class_id'])),line['xmin'],line['ymin'],str(float(line['xmax'])-float(line['xmin'])),str(float(line['ymax'])-float(line['ymin']))])+'\n')
    # break

input_file = csv.DictReader(open("./data_given/labels_trainval.csv"))

for line in tqdm(list(input_file)):
    # print(line)
    # print(line['LabelName'],classes_coded.index(line['LabelName']))
    with open('./data_given/test/labels/%s.txt'%line['frame'],'w') as f:
        f.write(' '.join([str(classes_coded.index(line['class_id'])),line['xmin'],line['ymin'],str(float(line['xmax'])-float(line['xmin'])),str(float(line['ymax'])-float(line['ymin']))])+'\n')
    # break