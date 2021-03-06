import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

from src.com.cv.PerspectiveTransform.Transform import order_points
from src.com.cv.SortContours.SortingContours import sort_contours, draw_text_in_center_contour


# https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/

# caculate mid point
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)


# load the image
image_path = "images/"
original_image = cv2.imread(image_path + "example_01.png")
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
(contours, _) = sort_contours(contours, method)
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

# with of reference object(in inches)
reference_object_width_by_metric = 1

# initialize 'pixels per metric' calibration variable
pixelsPerMetric = None

copy_original_image = original_image.copy()
# loop over the contours individually
for (index, contour) in enumerate(contours):
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(contour) < 100:
        continue
    # compute the rotated bounding box of the contour
    box = cv2.minAreaRect(contour)
    box_points = cv2.boxPoints(box)
    box_points = np.array(box_points, dtype="int")

    # order the points in the contour such that they appear in top-left, top-right, bottom-right, and bottom-left order
    rect = order_points(box_points)

    # loop over the ordered points and draw them
    for (x, y) in rect:
        cv2.circle(copy_original_image, (int(x), int(y)), 5, (0, 0, 255), -1)

    # show Ordered points
    window_name = 'Ordered Points'
    cv2.imshow(window_name, copy_original_image)
    cv2.waitKey()

    # draw the outline of the rotated bounding box
    cv2.drawContours(copy_original_image, [box_points], -1, (0, 255, 0), 2)

    # unpack the ordered bounding box
    (tl, tr, br, bl) = box_points
    # compute the midpoint between the top-left and top-right coordinates
    (tl_tr_X, tl_tr_Y) = midpoint(tl, tr)
    # compute the midpoint between bottom-left and bottom-right coordinates
    (bl_br_X, bl_br_Y) = midpoint(bl, br)
    # compute the midpoint between the top-left and bottom-left coordinates
    (tl_bl_X, tl_bl_Y) = midpoint(tl, bl)
    # the midpoint between the top-right and bottom-right
    (tr_br_X, tr_br_Y) = midpoint(tr, br)

    # draw the midpoints
    cv2.circle(copy_original_image, (int(tl_tr_X), int(tl_tr_Y)), 5, (255, 0, 0), -1)
    cv2.circle(copy_original_image, (int(bl_br_X), int(bl_br_Y)), 5, (255, 0, 0), -1)
    cv2.circle(copy_original_image, (int(tl_bl_X), int(tl_bl_Y)), 5, (255, 0, 0), -1)
    cv2.circle(copy_original_image, (int(tr_br_X), int(tr_br_Y)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(copy_original_image, (int(tl_tr_X), int(tl_tr_Y)), (int(bl_br_X), int(bl_br_Y)), (255, 0, 255), 2)
    cv2.line(copy_original_image, (int(tl_bl_X), int(tl_bl_Y)), (int(tr_br_X), int(tr_br_Y)), (255, 0, 255), 2)

    # show Ordered points
    window_name = 'Drawing Line'
    cv2.imshow(window_name, copy_original_image)
    cv2.waitKey()

    # compute the Euclidean distance between the midpoints
    distance_edge_1_by_pixel = distance.euclidean((tl_tr_X, tl_tr_Y), (bl_br_X, bl_br_Y))
    distance_edge_2_by_pixel = distance.euclidean((tl_bl_X, tl_bl_Y), (tr_br_X, tr_br_Y))

    # if the pixels per metric has not been initialized, then compute it as the ratio of pixels to supplied metric (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = distance_edge_2_by_pixel / reference_object_width_by_metric

    # compute the size of the current object
    distance_edge_1_by_metric = distance_edge_1_by_pixel / pixelsPerMetric
    distance_edge_2_by_metric = distance_edge_2_by_pixel / pixelsPerMetric
    # draw the object sizes on the image
    cv2.putText(copy_original_image, "{:.1f}in".format(distance_edge_1_by_metric), (int(tl_tr_X - 25), int(tl_tr_Y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(copy_original_image, "{:.1f}in".format(distance_edge_2_by_metric), (int(tr_br_X - 25), int(tr_br_Y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show the output image
    window_name = "Estimating object's size"
    cv2.imshow(window_name, copy_original_image)
    cv2.waitKey()
