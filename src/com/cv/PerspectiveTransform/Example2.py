from __future__ import print_function

import cv2
import imutils

# https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/

# load the image
image_path = "images/"
original_image = cv2.imread(image_path + "example_04.png")
# show original image
window_name = 'Original Image'
cv2.imshow(window_name, original_image)
cv2.waitKey()

# convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
# show Gray image
window_name = 'Gray Image'
cv2.imshow(window_name, gray_image)
cv2.waitKey()

# Blurring
blurring_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
# show Blurring image
window_name = 'Blurring Image'
cv2.imshow(window_name, blurring_image)
cv2.waitKey()

# find edges in the image
edged_image = cv2.Canny(blurring_image, 50, 100)
# show Edged image
window_name = 'Edged Image'
cv2.imshow(window_name, edged_image)
cv2.waitKey()

# perform a dilation + erosion to close gaps in between object edges
edged_image = cv2.dilate(edged_image, None, iterations=1)
# show dilation image
window_name = 'Dilation'
cv2.imshow(window_name, edged_image)
cv2.waitKey()

edged_image = cv2.erode(edged_image, None, iterations=1)
# show erosion image
window_name = 'Erosion'
cv2.imshow(window_name, edged_image)
cv2.waitKey()

# find contours in the edge map
contours = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# parsing the contours for various versions of OpenCV.
contours = imutils.grab_contours(contours)
