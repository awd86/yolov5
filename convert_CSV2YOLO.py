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

classes_coded = [1,2,3,4,5,6,7,8,9,10,11]
classes_names = ['biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green', 'trafficLight-GreenLeft', 'trafficLight-Red', 'trafficLight-RedLeft', 'trafficLight-Yellow', 'trafficLight-YellowLeft', 'truck']

print(classes_names.index(1))

# 601 classes, not 600...

input_file = csv.DictReader(open("./data_given/labels_train.csv"))

for line in tqdm(list(input_file)):
    print(line)
    print(line['class_id'],classes_coded.index(line['class_id']))
    with open('./data_given/train/labels/%s.txt'%line['frame'],'w') as f:
        f.write(','.join([str(classes_coded.index(line['class_id'])),line['XMin'],line['YMin'],str(float(line['XMax'])-float(line['XMin'])),str(float(line['XMax'])-float(line['YMin']))])+'\n')
    # break

input_file = csv.DictReader(open("./data_given/labels_trainval.csv"))

for line in tqdm(list(input_file)):
    # print(line)
    # print(line['LabelName'],classes_coded.index(line['LabelName']))
    with open('./data_given/test/labels/%s.txt'%line['frame'],'w') as f:
        f.write(','.join([str(classes_coded.index(line['class_id'])),line['XMin'],line['YMin'],str(float(line['XMax'])-float(line['XMin'])),str(float(line['XMax'])-float(line['YMin']))])+'\n')
    # break