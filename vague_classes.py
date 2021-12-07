import numpy as np


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
