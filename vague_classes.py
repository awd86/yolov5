import numpy as np
import os


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


# Doing this the correct way - based on mean bbox size by class label
def mean_class_bbox(label_dir,class_stops,class_names):
    # 'label_dir' is the directory containing the YOLO-format label files
    #       - if mustiple directories are provided as a list, it will iterate through all of them
    # 'class_stops' is a list specifying aggregated class endstops (where one stops and anther begins
    #       - if 3 numbers are specified in 'classes' then 4 classes will be created: [0,1,2,3]
    #       - values may be passes as type int or list. If type int, the int will be used as both [x,y]
    # 'class_names' is a list specifying names for the new aggregate classes.
    #       - needs to have 1 more value than class_stops (3 stops becomes 4 aggregate classes)

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

    # Handle 'class_stop' type and size
    if type(class_stops) == int:
        class_stops = [[x,x] for x in class_stops]
    elif not type(class_stops) == list:
        print(f"Formating error in 'class_stops' argument to 'mean_class_bbox()'\nshowing type: {type(class_stops)}")
        return

    # Handle 'label_dir' inputs
    if type(label_dir) == list:
        for x in label_dir:
            concat_files(x)
    elif type(label_dir) == str:
        concat_files(label_dir)
    else:
        print(f"'label_dir' needs to be passed into 'mean_class_bbox()' as a str or list:\ntype of 'label_dir': {type(label_dir)}")
        return  # exit the function because cannot be calculated

    # discover old classes, save as key in old_cls{}
    # track mean [x,y] dimensions, save as value in old_cls{}



def concat_files(dir):
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
# print(f'convert_strip list is {convert_strip}')

# Make a new class name list for the yaml
strip_classes = []
for k in keep_list:
    strip_classes.append(org_cls_names[k])
# print(strip_classes)
print(f'Stripped contains {len(strip_classes)} distinct classes')


###### create goldilocks dict ######
convert_gl = {}
for k in convert_strip.keys():
    if convert_strip[k]:
        convert_gl[k] = new_cls_num[k]
    else:
        convert_gl[k] = None
# print(convert_gl)
print(f'Goldilocks contains {sum(x is not None for x in list(convert_gl.values()))} sets aggregated into 3 classes')

###### create holyHandGrenade dict ######
convert_hhg = {}
for k in convert_gl.keys():
    if convert_gl[k] == 2:  # if 'just right'
        convert_hhg[k] = 1
    else:
        convert_hhg[k] = None
# print(convert_hhg)
print(f'HHG contains {sum(x is not None for x in list(convert_hhg.values()))} sets aggregated into 1 class')
