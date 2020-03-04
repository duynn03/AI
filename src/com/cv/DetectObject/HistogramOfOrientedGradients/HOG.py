from __future__ import print_function

import datetime

import cv2
import imutils

# https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/

# initializes the Histogram of Oriented Gradients descriptor
hog = cv2.HOGDescriptor()
# set the Support Vector Machine to be pre-trained pedestrian detector
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# load the image and
image_path = "DetectPedestrian/images/"
original_image = cv2.imread(image_path + "person_010.bmp")
# resize image
original_image = imutils.resize(original_image, width=min(400, original_image.shape[1]))
# show original image
window_name = 'Original Image'
cv2.imshow(window_name, original_image)
cv2.waitKey()

start = datetime.datetime.now()

# detect people in the image
# parameters
#       image pyramid with scale=1.05
#       a sliding window step size of (4, 4)  pixels in both the x and y direction
#           The size of the sliding window is fixed at 64 x 128 pixels
#           (http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
# return
#       bounding box (x, y)-coordinates of each person in the image
#       weights: the confidence value returned by the SVM for each detection
(boundingBoxs, weights) = hog.detectMultiScale(original_image, winStride=(8, 8), padding=(16, 16), scale=1.05,
                                               useMeanshiftGrouping=False)
print("[INFO] detection took: {}s".format((datetime.datetime.now() - start).total_seconds()))

# draw the original bounding boxes
HOG_SVM_image = original_image.copy()
for (x, y, w, h) in boundingBoxs:
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# show the output image
window_name = 'Detections'
cv2.imshow(window_name, original_image)
cv2.waitKey()
