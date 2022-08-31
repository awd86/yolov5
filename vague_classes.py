import numpy as np
import os
import warnings



org_cls_names = [
    'Fixed-wing Aircraft',
    'Small Aircraft',
    'Cargo Plane',
    'Helicopter',
    'Passenger Vehicle',
    'Small Car',
    'Bus',
    'Pickup Truck',
    'Utility Truck',
    'Truck',
    'Cargo Truck',
    'Truck w/Box',
    'Truck Tractor',
    'Trailer',
    'Truck w/Flatbed',
    'Truck w/Liquid',
    'Crane Truck',
    'Railway Vehicle',
    'Passenger Car',
    'Cargo Car',
    'Flat Car',
    'Tank car',
    'Locomotive',
    'Maritime Vessel', 'Motorboat', 'Sailboat', 'Tugboat', 'Barge',
    'Fishing Vessel', 'Ferry', 'Yacht', 'Container Ship', 'Oil Tanker', 'Engineering Vehicle', 'Tower crane',
    'Container Crane', 'Reach Stacker', 'Straddle Carrier',
    'Mobile Crane', 'Dump Truck', 'Haul Truck',
    'Scraper/Tractor',
    'Front loader/Bulldozer',
    'Excavator',
    'Cement Mixer',
    'Ground Grader',
    'Hut/Tent',
    'Shed',
    'Building',
    'Aircraft Hangar',
    'Damaged Building',
    'Facility',
    'Construction Site',
    'Vehicle Lot',
    'Helipad',
    'Storage Tank',
    'Shipping container lot',
    'Shipping Container',
    'Pylon',
    'Tower',
]
# print(org_cls_names)

'''
Manually Determine which classes are:
1. 'too small'    < 7 m
2. 'just right'   7 <= l <= 15 m
3. 'too large'    > 15 m
'''

# Manually-created re-labeling scheme
new_cls_num = [
         1,
         1,
         3,
         1,
         1,
         1,
         2,
         2,
         2,
         2,
         2,
         2,
         2,
         2,
         2,
         2,
         2,
         2,
         1,
         2,
         2,
         2,
         2,
         3,
         1,
         1,
         3,
         3,
         3,
         3,
         3,
         3,
         3,
         2,
         3,
         3,
         3,
         3,
         2,
         2,
         2,
         2,
         2,
         2,
         2,
         2,
         1,
         1,
         3,
         3,
         3,
         3,
         3,
         3,
         3,
         3,
         3,
         2,
         3,
         3,
]
new_cls_num = [x-1 for x in new_cls_num]  #reset to 0-biased

# TODO implement 'aggregate'
# Doing this the correct way - based on mean bbox size by class label
def aggregate(label_dir,method,**kwargs):
    # 'label_dir' is the directory containing the YOLO-format label files
    #       - if multiple directories are provided as a list, it will iterate through all of them
    # 'method' specifies the way in which to aggregate
    #       - 'xMean' will find the mean of x_width in YOLO format across all label instances in label_dir
    # kwargs:
    # 'class_stops' (method = xMean) is a list of int specifying aggregated class endstops (where one stops and another begins)
    #       - if 3 numbers are specified in 'classes' then 4 classes will be created: [0,1,2,3]
    #       - values must be passes as type int. There is no way to prioritize x or y if they are different.
    #       - #### the program will only use 'x' values ####
    # 'class_names' (method = xMean) is a list specifying names for the new aggregate classes.
    #       - needs to have 1 more value than class_stops (3 stops becomes 4 aggregate classes)

    # Handle 'label_dir' inputs
    if type(label_dir) == list:
        for x in label_dir:
            label_files = concat_files(x)
    elif type(label_dir) == str:
        label_files = concat_files(label_dir)
    else:
        print(f"'label_dir' needs to be passed into 'mean_class_bbox()' as a str or list:\ntype of 'label_dir': {type(label_dir)}")
        return  # exit the function because cannot be calculated

    # Handle method and kwargs
    if method == 'xMean':

        # Check 'class_name' and 'class_stop' exist
        try:
            class_stops = kwargs['class_stops']
            class_names = kwargs['class_names']
        except:
            print(f"Insufficent specification for method:xMean. Requires 'class_stops' and 'class_names'")

        # Check 'class_name' and 'class_stop' sizes
        try:
            if not len(class_names)%len(class_stops) == 1:
                print(f"'class_names' needs to have 1 more value than 'class_stops'\ne.g. 3 stops creates 4 classes\n")
                print(f'{len(class_names)} class names provided, {len(class_stops)} class stops provided')
                return  # exit the entire function after this error b/c 'mean_class_bbox' cannot be completed
        except:
            print("Formating error in 'class_stops' or 'class_names' argument to 'mean_class_bbox()'")
            print(f"please check that there are type 'list':\ntype of 'class_names': {type(class_names)}\ntype of 'class_stops': {type(class_stops)}")
            return

        # Handle 'class_stop' type and size  #### only using x dimension ####
        if type(class_stops[0]) == int or float:
            # class_stops = [[x,x] for x in class_stops]
            # 'class_stops' can only accept x-values for xMean (unless upgraded)
            pass
        else:
            print(f"Formating error in 'class_stops' argument to 'mean_class_bbox(). Only accepts list of int or float.'\nshowing type of class_stops[0]: {type(class_stops[0])}")
            return

        # Find xMean
        old_cls,old_xMean = label_Mean(label_files,3)
        print(f'Original Class Instances:\n{old_cls}')
        print(f'Average x_width by Class:\n{old_xMean}')

        # Re-Classify by xMean with 'class_stops'
        new_cls = {}
        class_stops = [0] + class_stops + [math.inf]  # properly buffer the list of stops for >v>=

        for cls in range(len(class_stops)-1):
            new_cls.update({k:cls for (k,v) in old_xMean.items() if class_stops[cls + 1] > v >= class_stops[cls]})  # dictionary comprehension

        print(f'Aggregated class substitutions:\n{new_cls}')

        # Create dict associating class_names with aggregated classes
        old_names = {k: class_names[new_cls[k]] for k in old_cls.keys()}
        print(f"Old name associations are:\n{old_names}")
        new_names = {k:class_names[k] for k in range(len(class_names))}
        print(f"New name associations are:\n{new_names}")


        return new_cls  # the only required return is the old_class:new_class associations in 'new_cls'


    else:  # method and kwargs
        print(f'Aggregation method {method} not specified')

def label_Mean(label_files, col):
    # Based on a list 'label_files', this opens the files and reads class and x_width data
    # Recorded data is then stored as class_num:[sum(x_width),count(x_width)]
    # Returns old_xMean = class_num:(sum(x_width)/count(x_width)); old_cls = class_num:count(x_width)

    # Discover old class names, save as key in old_cls{}
    old_cls = {}
    for _label in label_files:

        # load labels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                labels = np.genfromtxt(_label, delimiter=None, )
            except:
                continue

            if len(labels) == 0:  # if no labels, skip the remainder
                continue

        labels = np.atleast_2d(labels)

        try:
            for i in range(len(labels[:, 0])):  # iterates through all entries in 'labels'
                i = int(i)
                cls = labels[i, 0]
                if not cls in old_cls.keys():  # check for cls present in old_cls
                    old_cls[int(cls)] = [labels[i, col],
                                         1]  # x_width; method is the column of the label file (usually 3=x_width)
                else:
                    old_cls[cls] = [old_cls[cls][0] + labels[i, col],
                                    old_cls[cls][1] + 1]  # store [sum,inst] for later division
        except:
            print(f"labels failed:\n {labels}")

    # _old_cls = pd.DataFrame.from_dict(old_cls,orient='index', columns=['sum x_width','instances'])
    # print(f'Original Classes:\n{_old_cls}')

    # Calculates the Mean
    old_xMean = {}
    for cls in old_cls.keys():
        old_xMean[cls] = old_cls[cls][0] / old_cls[cls][1]  # find the mean for each class
        old_cls[cls] = old_cls[cls][1]  # save the instances
    # print(f'Original Class Instances:\n{old_cls}')
    # print(f'Average x_width by Class:\n{old_xMean}')
    #
    return (old_cls, old_xMean)

def concat_files(dir):
    # creates a list of the names of all label files (all files, really) in 'dir' that end in '*.txt'
    label_files = []
    try:
        for file in os.listdir(path=f'{dir}'):  # ["[image number].tif",...]
            if file.endswith('.txt'):
                label_files.append(f'{dir}/{file}')
    except:
        print("Check 'label_dir' argument to 'mean_class_bbox()'")
        return  # exit the function because cannot be calculated
    return label_files


new_cls_name = new_cls_num.copy()
org_cls_num = []

for i in range(len(new_cls_num)):
    org_cls_num.append(i)
    if new_cls_num[i] == 1:
        new_cls_name[i] = 'too small'
    elif new_cls_num[i] == 2:
        new_cls_name[i] = 'just right'
    elif new_cls_num[i] == 3:
        new_cls_name[i] = 'too big'
    elif new_cls_num[i] == None:
        new_cls_name[i] = None
    else:
        print(f'{new_cls_num[i]} not recognized')

conversion = np.array([org_cls_num,new_cls_num,org_cls_names,new_cls_name]).T

# print(conversion)
#
# print(new_cls_name)

# What to keep
keep_list = list(range(4,23))+list(range(39,48))+[55,57]

###### Create stripped dict ######
convert_strip = {}
i = 0
for k in org_cls_num:
    if k in keep_list:
        convert_strip[k] = i
        i +=1
    else:
        convert_strip[k] = None
#print(f'Stripper:   {convert_strip}')
for k , v in convert_strip.items(): # iterating freqa dictionary
    print(f'{k:<4} {v}')

# Make a new class name list for the yaml
strip_classes = []
for k in keep_list:
    strip_classes.append(org_cls_names[k])
#print(strip_classes)
print(f'Stripped contains {len(strip_classes)} distinct classes')


###### create goldilocks dict ######
convert_gl = {}
for k in convert_strip.keys():
    if convert_strip[k]:
        convert_gl[k] = new_cls_num[k]
    else:
        convert_gl[k] = None
#print(f'Goldilocks: {convert_gl}')
for k , v in convert_gl.items(): # iterating freqa dictionary
    print(f'{k:<4} {v}')
print(f'Goldilocks contains {sum(x is not None for x in list(convert_gl.values()))} sets aggregated into 3 classes')


###### create holyHandGrenade dict ######
convert_hhg = {}
for k in convert_gl.keys():
    if convert_gl[k] == 2:  # if 'just right'
        convert_hhg[k] = 1
    else:
        convert_hhg[k] = None
#print(f'HHG:        {convert_hhg}')
for k , v in convert_hhg.items(): # iterating freqa dictionary
    print(f'{k:<4} {v}')
print(f'HHG contains {sum(x is not None for x in list(convert_hhg.values()))} sets aggregated into 1 class')
