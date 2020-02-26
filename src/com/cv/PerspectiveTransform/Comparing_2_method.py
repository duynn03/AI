from __future__ import print_function

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

from src.com.cv.PerspectiveTransform.Transform import order_points_old, order_points
from src.com.cv.SortContours.SortingContours import sort_contours, draw_text_in_center_contour

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

# draw text in center contours in origin image
unsorted_original_image = original_image.copy()
# loop over the (unsorted) contours and draw contour
for (index, contour) in enumerate(contours):
    unsorted_original_image = draw_text_in_center_contour(unsorted_original_image, contour, index)

# sort the contours according to the provided method
method = "left-to-right"
(contours, boundingBoxes) = sort_contours(contours, method)
sorted_original_image = original_image.copy()
# loop over the (now sorted) contours and draw them
for (index, contour) in enumerate(contours):
    draw_text_in_center_contour(sorted_original_image, contour, index)

# show result
fig = plt.figure("Sorting Contours by " + method)
images = ("Unsorted", unsorted_original_image), ("Sorted", sorted_original_image)
# show the image
for (index, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 2, index + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()

method_1_original_image = original_image.copy()
method_2_original_image = original_image.copy()
# loop over the contours individually
for (index, contour) in enumerate(contours):
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(contour) < 100:
        continue

    # compute the rotated bounding box of the contour
    box = cv2.minAreaRect(contour)
    box_points = cv2.boxPoints(box)
    box_points = np.array(box_points, dtype="int")
    # show the original coordinates
    print("Object #{}:".format(index + 1))
    print(box_points)

    # draw the contours
    cv2.drawContours(method_1_original_image, [box_points], -1, (0, 255, 0), 2)
    cv2.drawContours(method_2_original_image, [box_points], -1, (0, 255, 0), 2)

    # show compare contour sorting
    fig = plt.figure("Contours")
    images = ("Method 1 Contours", method_1_original_image), ("Method 2 Contours", method_2_original_image)
    # show the image
    for (i, (name, image)) in enumerate(images):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.set_title(name)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
    # show the figure
    plt.show()

    # order the points in the contour such that they appear in top-left, top-right, bottom-right, and bottom-left order
    # draw the outline of the rotated bounding box
    rect_method_1 = order_points_old(box_points)
    # print rect
    print("Order points by method 1: \n", rect_method_1.astype("int"))

    # top-left: red, top-right: purple, bottom-right: blue, bottom-left: teal.
    colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

    # loop over the original points and draw them
    for ((x, y), color) in zip(rect_method_1, colors):
        cv2.circle(method_1_original_image, (int(x), int(y)), 5, color, -1)

    # draw the object num at the top-left corner
    cv2.putText(method_1_original_image, "Object #{}".format(index + 1),
                (int(rect_method_1[0][0] - 15), int(rect_method_1[0][1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)

    # method 2
    rect_method_2 = order_points(box_points)
    # print rect
    print("Order points by method 2: \n", rect_method_2.astype("int"))

    # loop over the original points and draw them
    for ((x, y), color) in zip(rect_method_2, colors):
        cv2.circle(method_2_original_image, (int(x), int(y)), 5, color, -1)

    # draw the object num at the top-left corner
    cv2.putText(method_2_original_image, "Object #{}".format(index + 1),
                (int(rect_method_2[0][0] - 15), int(rect_method_2[0][1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)

    # show compare 2 method
    fig = plt.figure("Step " + str(index + 1) + ":")
    images = ("Method 1", method_1_original_image), ("Method 2", method_2_original_image)
    # show the image
    for (i, (name, image)) in enumerate(images):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.set_title(name)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
    # show the figure
    plt.show()
