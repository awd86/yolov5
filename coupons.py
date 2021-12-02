#####################################################
#   Alex Denton, 1 Dec 2021, AE4824                 #
#   coupons.py contains 'clipper' and 'stitcher'    #
#   'clipper' breaks an image/labels into tiles     #
#   'stitcher' reverses the process                 #
#####################################################


# Import packages
import cv2
import numpy as np
from os import mkdir, path
from pathlib import Path
from math import ceil
from tqdm import tqdm

def clipper(**kwargs):

    # 'kwargs': (these are the only valid inputs)
    #   'image': **required** relative path to source image
    #   'frame': **required** single value is x&y, otherwise list [x,y]
    #   'labels': label file (with path if not same as image)
    #       'try' and warn if not found
    #       default to searching current, parent, and parent/labels/
    #   'step' OR 'overlap': single value is x&y, otherwise list [x,y]
    #       throw error if both methods provided
    #       default to 'try' calculating off largest labeled bbox
    #       if neither and no labels, use step = 0.75*frame
    #   'auto_step': coefficient for multiplying largest bbox dimensions
    #       * incompatible with 'step' or 'overlap'
    #       default value is 1.2
    #   'fill': what to do with remainder in last row/column (applies same to both)
    #       * only applies if 'step' or 'overlap' provided
    #       (calculated off labeled bbox doesn't require 'fill' - error if both)
    #       options:
    #           '0': black-fill *default
    #           '1': right-justify
    #           '2': wrap (same row/column)
    #   'geo': [latitude, longitude] of all four corners of original [bottom left, bottom right, top right, top left]
    #       dd.mm.ss.dddd or dd.dddddddd
    #       will fill zeros

    ###### Check for required inputs #####
    if not 'image' in kwargs:
        print(f'No image provided to clipper!')
        return  # quit early
    else:
        image = kwargs['image']
        if not path.exists(image):  # check for existance
            print(f"Image '{image}' not found. Check path.")
            return  # quit early

    if not 'frame' in kwargs:
        print(f'No frame dimensions provided to clipper. Defaulting to 512x512px')
        frame = [512,512]
    else:
        frame = kwargs['frame']


    ###### Load and Measure Image #####

    # Load File
    img = cv2.imread(image)
    X, Y, L = img.shape
    #print(f'Image dimensions are X={X}, Y={Y}')
    #cv2.imshow("original", img)

    # Determine box size
    if len(frame) == 2:  # 2 is desired
        xf,yf = frame[0:2]
    elif len(frame) == 1:  # if single dimension, make it square
        xf = frame[0]
        yf = xf
    else:
        print(f"clipper only supports 2D frames, using only the first two")
        xf,yf = frame[0:2]


    ###### Folder Management #####

    # Determine folder to parent 'clipped_images'
    img_dir = Path(image).parent.absolute()
    if 'images' == str(img_dir).split('/')[-1]:  # only finds lowest directory labeled 'images/'
        par_dir = img_dir.parent.absolute()  # go up one level so that "clipped_images/" is parallel to "images/"
    else:
        par_dir = img_dir

    # Create the 'clipped_images' folder
    img_dir = Path(f'{par_dir}/clipped_images')
    if not path.exists(img_dir):
        mkdir(path.join(par_dir,'clipped_images'))

    # Create the 'clipped_labels' folder if applicable
    if 'labels' in kwargs:
        lbl_dir = Path(f'{par_dir}/clipped_labels')
        if not path.exists(lbl_dir):
            mkdir(path.join(par_dir, 'clipped_labels'))


    ###### Load Labels #####

    # Find Label File
    if 'labels' in kwargs:
        # Load labels as numpy array
        labels = kwargs['labels']  # path and label.txt file
        if not path.exists(labels):
            print(f'Specified label file does not exist. \n\tCheck path: {labels}')
        labels = np.genfromtxt(labels,delimiter=' ')
        if type(labels[0,1]) == str:
            print('Label file is not in YOLO format!')
            return

        # Convert from Normalized (0 to 1) x_ctr y_ctr x_width y_height
        #           to Absolute (min, max) X_ctr Y_ctr X_width Y_height
        labels[:, 1] *= X  # x_ctr
        labels[:, 2] *= Y  # y_ctr
        labels[:, 3] *= X  # x_width
        labels[:, 4] *= Y  # y_height
        #print(f"The absolute label dimensions are:\n{labels}")


    ##### Determine step size #####

    # Determine Step Size
    if ('step' in kwargs or 'overlap' in kwargs) and 'auto_step' in kwargs:
        print("'auto_step' is incompatible with 'step' and 'overlap' and is not being used")

    if 'step' in kwargs:  # step is the value that will be used
        if type(kwargs['step']) is not list:
            step = kwargs['step']
            step = [step,step]  # use same step for x & y
        else:
            step = list(kwargs['step'])  #accepts list[] or tuple()

        if len(step) > 2:
            step = step[0:2]  #only use first 2 entries
            print(f"Provided 'step' is too long. Using step={step}")

        if 'overlap' in kwargs:  #overdedfined
            print(f"Both 'step' and 'overlap' provided. Only using step={step}")

    else:
        if 'overlap' in kwargs:
            if type(kwargs['overlap']) is not list:
                overlap = kwargs['overlap']
                overlap = [overlap, overlap]  # use same step for x & y
            else:
                overlap = list(kwargs['overlap'])  # accepts list[] or tuple()

            if len(overlap) > 2:
                overlap = overlap[0:2]  # only use first 2 entries
                print(f"Provided 'overlap' is too long. Using overlap={overlap}")
            # convert to 'step'
            step = np.subtract(frame,overlap)

        elif 'labels' in kwargs:  # determine overlap based on max bbox (1.2*)
            w_max = np.max(a=labels[:,3],axis=0)  # find max bbox width in absolute px
            h_max = np.max(a=labels[:,4],axis=0)
            #print(f"max bbox dimensions are {[w_max,h_max]}")

            # Calculate the minimum overlap
            if 'auto_step' in kwargs:
                overlap = np.multiply([w_max,h_max],kwargs['auto_step'])  # handles 1 or 2D auto_step
            else:
                overlap = np.multiply([w_max, h_max], 1.2).astype(int)
            step = np.subtract(frame,overlap)

            # Adjust to perfect fit
            img_dim = [X,Y]
            for n in [0,1]:
                rem = img_dim[n]-frame[n]
                if not rem % step[n] == 0:  # if there is any overlap, need one more frame
                    ct = rem // step[n] +1  # make one more column
                    step[n] = ceil(rem / ct )  # round up to nearest int  # TODO finish end cell treatments

            print(f"'auto_step' has calculated an optimal multipliers of {[(frame[0]-step[0])/w_max,(frame[1]-step[1])/h_max]}")

        else:
            step = np.multiply(0.75,frame)  # ensure integer (no partial pixels)
            step = np.ceil(step).astype(int)  # roudn up to ensure overflow (no pixels dropped)
            print(f'No step size provided. Using step=0.75*frame : step={step}')


    print(f'step={step}')


    ##### Clip Images (and Labels, if applicable) #####

    cropped_image = img[0:frame[0], 0:frame[1]]

    # Display cropped image
    #cv2.imshow("cropped", cropped_image)

    # Save the cropped image
    #cv2.imwrite(f"{src_dir}/Cropped Image.jpg", cropped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # TODO rename 'images' to 'original_images' and 'clipped_images' to 'images'

###### Testing ######
img = 'data_xView/10.jpg'
lbl = 'data_xView/10.txt'

clipper(image=img, frame=[512,512], labels=lbl,)
