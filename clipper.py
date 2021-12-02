# Import packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

cv2.namedWindow('displaymywindows', cv2.WINDOW_NORMAL)

img = cv2.imread('data_xView/10.jpg')
print(img.shape) # Print image shape
cv2.imshow("original", img)
# plt.imshow(img)

# # Cropping an image
# cropped_image = img[80:280, 150:330]
#
# # Display cropped image
# cv2.imshow("cropped", cropped_image)
#
# # Save the cropped image
# cv2.imwrite("Cropped Image.jpg", cropped_image)
#
cv2.waitKey(0)
cv2.destroyAllWindows()