from PIL import Image
import numpy as np
import cv2
# def make_square(im, size=[512,512], fill_color=(0, 0, 0, 0)):
#     # x, y = im.size
#     # size = max(min_size, x, y)
#     # new_im = Image.new('RGBA', (size, size), fill_color)
#     new_im = Image.new('RGBA', size, fill_color)
#     new_im.paste(im, (0, 0))
#     return new_im

def blackfill(img, frame, color=(0,0,0,0)):
    new_img = np.zeros([frame[0],frame[1],img.shape[2]],np.uint8)
    new_img[0:img.shape[0],0:img.shape[1]] = img
    return new_img

frame = [512,512]
#test_image = Image.open('data_xView/images/10_008_007.jpg')
test_image = cv2.imread('data_xView/images/10_008_007.jpg')
new_image = blackfill(test_image,frame)
new_image.imshow()