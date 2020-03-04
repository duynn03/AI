from __future__ import print_function

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from imutils.object_detection import non_max_suppression

# load the image
image_path = "images/"

# initializes the Histogram of Oriented Gradients descriptor
hog = cv2.HOGDescriptor()
# set the Support Vector Machine to be pre-trained pedestrian detector
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
for original_image in paths.list_images(image_path):
    # load the image and resize it to reduce detection time and improve detection accuracy
    original_image = cv2.imread(original_image)
    original_image = imutils.resize(original_image, width=min(400, original_image.shape[1]))
    # show original image
    window_name = 'Original Image'
    cv2.imshow(window_name, original_image)
    cv2.waitKey()

    # detect people in the image
    # parameters
    #       image pyramid with scale=1.05
    #       a sliding window step size of (4, 4)  pixels in both the x and y direction
    #           The size of the sliding window is fixed at 64 x 128 pixels
    #           (http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
    # return
    #       bounding box (x, y)-coordinates of each person in the image
    #       weights: the confidence value returned by the SVM for each detection
    #
    (boundingBoxs, weights) = hog.detectMultiScale(original_image, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    HOG_SVM_image = original_image.copy()
    for (x, y, w, h) in boundingBoxs:
        cv2.rectangle(HOG_SVM_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # show image
    window_name = 'Apply HOG and SVN'
    cv2.imshow(window_name, HOG_SVM_image)
    cv2.waitKey()

    # apply non-maximum suppression to the bounding boxes using a fairly large overlap threshold to try to maintain overlapping boxes that are still people
    boundingBoxs = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boundingBoxs])
    keeping_boundingBoxs = non_max_suppression(boundingBoxs, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    NMS_image = original_image.copy()
    for (start_X, start_Y, end_X, end_Y) in keeping_boundingBoxs:
        cv2.rectangle(NMS_image, (start_X, start_Y), (end_X, end_Y), (0, 255, 0), 2)

    # show image
    window_name = 'Apply non-maximum suppression'
    cv2.imshow(window_name, NMS_image)
    cv2.waitKey()

    # show Result
    fig = plt.figure("Result")
    images = ("Before", original_image), ("After Detecting ", NMS_image)
    for (i, (name, image)) in enumerate(images):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.set_title(name)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
    # show the figure
    plt.show()
