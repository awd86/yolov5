import numpy as np


names = ['Fixed-wing Aircraft', 'Small Aircraft', 'Cargo Plane', 'Helicopter', 'Passenger Vehicle', 'Small Car', 'Bus',
        'Pickup Truck', 'Utility Truck', 'Truck', 'Cargo Truck', 'Truck w/Box', 'Truck Tractor', 'Trailer',
        'Truck w/Flatbed', 'Truck w/Liquid', 'Crane Truck', 'Railway Vehicle', 'Passenger Car', 'Cargo Car',
        'Flat Car', 'Tank car', 'Locomotive', 'Maritime Vessel', 'Motorboat', 'Sailboat', 'Tugboat', 'Barge',
        'Fishing Vessel', 'Ferry', 'Yacht', 'Container Ship', 'Oil Tanker', 'Engineering Vehicle', 'Tower crane',
        'Container Crane', 'Reach Stacker', 'Straddle Carrier', 'Mobile Crane', 'Dump Truck', 'Haul Truck',
        'Scraper/Tractor', 'Front loader/Bulldozer', 'Excavator', 'Cement Mixer', 'Ground Grader', 'Hut/Tent', 'Shed',
        'Building', 'Aircraft Hangar', 'Damaged Building', 'Facility', 'Construction Site', 'Vehicle Lot', 'Helipad',
        'Storage Tank', 'Shipping container lot', 'Shipping Container', 'Pylon', 'Tower']

# print(names)

'''
Manually Determine which classes are:
1. 'too small'    < 7 m
2. 'just right'   7 <= l <= 15 m
3. 'too large'    > 15 m
'''

names2 = [
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

names3 = names2.copy()

for i in range(len(names2)):
    blah[i]=i
        if names2[i] == 1:
                names3[i] = 'too small'
        elif names2[i] == 2:
                names3[i] = 'just right'
        else:
                names3[i] = 'too big'

conversion = np.array([names2,names,names3]).T

# print(conversion)
#
# print(names3)
convert = {}
for k in range(len(names2)):
    convert[k]=names2[k]

#print(convert)