#####################################################
#   Alex Denton, 1 Dec 2021, AE4824                 #
#   coupons.py contains 'clipper' and 'stitcher'    #
#   'clipper' breaks an image/labels into tiles     #
#   'stitcher' reverses the process                 #
#####################################################


# Import packages
import cv2
from PIL import Image  # used in 'black_fill'
import numpy as np
import os
from pathlib import Path
from math import ceil
from tqdm import tqdm

# TODO create 'batch_clipper' function to pull an entire directory (and calculate auto_step)
# could implement a 'batch' kwarg to prevent repeating input validation

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
        if not os.path.exists(image):  # check for existance
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
    img_dim = img.shape[0:2]  # ignore depth
    #X, Y, L = img.shape
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


    ###### Load Labels #####

    # Find Label File
    if 'labels' in kwargs:
        # Load labels as numpy array
        labels = kwargs['labels']  # path and label.txt file
        if not os.path.exists(labels):
            print(f'Specified label file does not exist. \n\tCheck path: {labels}')
        labels = np.genfromtxt(labels,delimiter=' ')
        if type(labels[0,1]) == str:
            print('Label file is not in YOLO format!')
            return

        # Convert from Normalized (0 to 1) x_ctr y_ctr x_width y_height
        #           to Absolute (min, max) X_ctr Y_ctr X_width Y_height
        labels[:, 1] *= img_dim[0]  # x_ctr
        labels[:, 2] *= img_dim[1]  # y_ctr
        labels[:, 3] *= img_dim[0]  # x_width
        labels[:, 4] *= img_dim[1]  # y_height
        #print(f"The absolute label dimensions are:\n{labels}")

        # Convert alternate labels_minMax with x_min, x_Max, y_min, y_Max
        labels_minMax = labels.copy()
        labels_sub = labels.copy()
        #print(f"labels is:\n{labels[0:4,:]}")

        labels_sub[:,3] /= 2  # half width
        labels_sub[:,4] /= 2  # half height
        #print(f"labels_sub is:\n{labels_sub[0:4,:]}")

        labels_minMax[:, 3] = labels_minMax[:, 2]  # y_ctr
        labels_minMax[:, 2] = labels_minMax[:, 1]  # x_ctr
        labels_minMax[:, 4] = labels_minMax[:, 3]  # y_ctr

        labels_minMax[:, 1] -= labels_sub[:,3]  # label's x_min
        labels_minMax[:, 2] += labels_sub[:,3]  # label's x_Max
        labels_minMax[:, 3] -= labels_sub[:,4]  # label's y_min
        labels_minMax[:, 4] += labels_sub[:,4]  # label's y_Max

        #print(f"The labels_minMax are:\n{labels_minMax[0:4,:]}")
        del labels_sub  # manual trash collecting

    ##### Batch Management #####
    try:
        batch = kwargs['batch']
    except:
        batch = False

    ###### Folder Management #####

    # TODO fix batch handling

    # Determine folder to parent 'clipped_images'
    img_dir = Path(image).parent.absolute()
    if 'images' == str(img_dir).split('/')[-1]:  # only finds lowest directory labeled 'images/'
        par_dir = img_dir.parent.absolute()  # go up one level so that "clipped_images/" is parallel to "images/"
    else:
        par_dir = img_dir

    # Create the 'clipped_images' folder
    img_dir = Path(f'{par_dir}/clipped_images')
    if not os.path.exists(img_dir):
        os.mkdir(os.path.join(par_dir,'clipped_images'))

    # Create the 'clipped_labels' folder if applicable
    if 'labels' in kwargs:
        lbl_dir = Path(f'{par_dir}/clipped_labels')
        if not os.path.exists(lbl_dir):
            os.mkdir(os.path.join(par_dir, 'clipped_labels'))

        #i += 1  # only do this once


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

    img_name = image.split('/')[-1].split('.')[0]  # drops all suffixes and loses file type
    img_type = image.split('/')[-1].split('.')[-1]

    steps = np.ceil( np.add( np.divide( np.subtract(img_dim,frame), step), 1) )

    for y_trav in tqdm(range(steps.astype(int)[1])):
        for x_trav in range(steps.astype(int)[0]):
            _x = x_trav*step[0]
            _y = y_trav*step[1]

            # Clip Images
            clip = img[_x:_x+frame[0], _y:_y+frame[1]]

            # Handle partial images
            if not clip.shape[:2] == frame:  # if the image isn't filling the frame
                try:
                    fill = kwargs['fill']
                except:
                    fill = 0

                if not fill == 0:
                    print('Only blackfill method has been built. Using blackfill.')
                    fill = 0
                else:
                    clip = blackfill(clip, frame)


            # Write Image
            cv2.imwrite(f"{img_dir}/{img_name}_{y_trav:03}_{x_trav:03}.{img_type}", clip)

            try:  # won't execute if no labels given
                clip_labels = np.array([0, 0, 0, 0, 0])  # pre-allocate array
                for row in range(len(labels[:,0])):  # count number of rows

                    # Check for label containment in clip (based on absolute dimension of original image)
                    #   structure of a 'labels_minMax' row is [class x_min x_Max y_min y_Max]
                    if labels_minMax[row,1] >= _x and labels_minMax[row,2] <= (_x+frame[0]):  # x containment
                        if labels_minMax[row,3] >= _y and labels_minMax[row,4] <= (_y+frame[1]):  # y containment (effectively another 'and' statement)

                            # Pull in the entire row with absolute min/max
                            label_rel = labels_minMax[row,:]

                            # Set relative to frame
                            label_rel[1] -= _x  # x_min
                            label_rel[2] -= _x  # x_Max
                            label_rel[3] -= _y  # y_min
                            label_rel[4] -= _y  # y_Max
                            #print(f"label_rel {y_trav:03}_{x_trav:03} relative to frame is:\n{label_rel}")

                            # YOLO format: [class x_ctr y_ctr x_width y_height]
                            x_ctr = (label_rel[1]+label_rel[2]) /2
                            y_ctr = (label_rel[3]+label_rel[4]) /2
                            x_width = label_rel[2]-label_rel[1]
                            y_height = label_rel[4]-label_rel[3]

                            # Normalize to frame
                            label_rel[1] = x_ctr/frame[0]
                            label_rel[2] = y_ctr/frame[1]
                            label_rel[3] = x_width/frame[0]
                            label_rel[4] = y_height/frame[1]
                            #print(f"label_rel {y_trav:03}_{x_trav:03} in norm YOLO is:\n{label_rel}")

                            if clip_labels.all() == 0:  # if no data yet
                                clip_labels = labels[row,:]  # first row
                                #print(f'first row filled:\n{clip_labels}')
                            else:
                                clip_labels = np.vstack([clip_labels, labels[row, :]])
                                #print(f'row added:\n{clip_labels}')
                            #print(f"clip_labels {y_trav:03}_{x_trav:03} is:\n{clip_labels}")

                # If no labels, make the file blank (YOLO format)
                if clip_labels.all() == 0:
                    clip_labels = np.array([])

                # Write Label file
                np.savetxt(f"{lbl_dir}/{img_name}_{y_trav:03}_{x_trav:03}.txt", np.atleast_2d(clip_labels), delimiter=' ', newline='\n', encoding=None)

            except:
                pass

    print(f"Created {int(steps[0]*steps[1])} new clipped images.")



def rename_folders(par_dir):
    # Rename 'images' folders
    try:
        os.rename(f'{par_dir}/images',f'{par_dir}/original_images')
    except:
        print("No original 'labels' found found in parent directory. Renaming 'clipped_images' to 'images'")

    try:
        os.rename(f'{par_dir}/clipped_images',f'{par_dir}/images')
    except:
        print("No 'clipped_images' folder found to rename.")

    # Rename 'labels' folders
    try:  # there may not be a labels file
        os.path.exists(f'{par_dir}/labels')
        try:
            os.rename(f'{par_dir}/labels', f'{par_dir}/original_labels')
            os.rename(f'{par_dir}/clipped_labels',f'{par_dir}/labels')
        except:
            print("No 'labels' folder found.")
    except:
        pass


# TODO create 'stitcher' to recombine coupons and labels


def blackfill(img, frame, color=(0,0,0,0)):
    new_img = np.zeros([frame[0],frame[1],img.shape[2]],np.uint8)
    new_img[0:img.shape[0],0:img.shape[1]] = img
    return new_img


# TODO 'wrapfill' funciton needs to adjust labels
def wrapfill(img,wrap_img):
    wrap_img[0:img.shape[0],0:img.shape[1]] = img
    return wrap_img


def batch_clip(img_dir, **kwargs):
    # kwargs includes lbl_dir
    images = []
    for file in os.listdir(path=f'{img_dir}'):  # ["[image number].tif",...]
        if file.endswith('.jpg' or '.tif'):
            images.append(f'{img_dir}/{file}')

    try:
        lbl_dir = kwargs['lbl_dir']
        kwargs.pop('lbl_dir')  # it's been used, now remove it
    except:
        src_dir = Path(img_dir).parent.absolute()
        try:
            lbl_dir = os.path(f"{src_dir}/labels")
        except:
            print('No label folder found.')
            lbl_dir = False

    for img in images:
        kwargs['image'] = img

        if lbl_dir:
            lbl = str(img).split('/')[-1].split('.')[0]  # separate out the file name
            lbl = Path(f'{lbl_dir}/{lbl}.txt')
            try:
                os.path.exists(lbl)
                kwargs['labels'] = lbl
            except:
                print(f"Label file {lbl} not found")
                del lbl

        print(f'\nPassing this to clipper:\n{kwargs}\n')
        clipper(**kwargs)


###### Testing ######
img = 'data_xView/10.jpg'
lbl = 'data_xView/10.txt'

#clipper(image=img, frame=[512,512], labels=lbl)
batch_clip(img_dir='data_xView', lbl_dir = 'data_xView', frame = [512,512])