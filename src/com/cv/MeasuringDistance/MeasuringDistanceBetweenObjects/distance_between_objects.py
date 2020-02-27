import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

from src.com.cv.PerspectiveTransform.Transform import order_points
from src.com.cv.SortContours.SortingContours import sort_contours, draw_text_in_center_contour


# https://www.pyimagesearch.com/2016/04/04/measuring-distance-between-objects-in-an-image-with-opencv/

# caculate mid point
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)


# load the image
image_path = "images/"
original_image = cv2.imread(image_path + "example_03.png")
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

# reference_object is a 3-tuple (including bounding box, centroid, and pixels-per-metric value of the reference object)
reference_object = None

# with of reference object(in inches)
reference_object_width_by_metric = 1

# loop over the contours individually
for (index, contour) in enumerate(contours):
    copy_original_image = original_image.copy()

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

    # compute the center of the bounding box
    center_X = np.average(rect[:, 0])
    center_Y = np.average(rect[:, 1])

    # draw center point
    cv2.circle(copy_original_image, (int(center_X), int(center_Y)), 5, (0, 0, 255), -1)
    # show center points
    window_name = 'Center Points'
    cv2.imshow(window_name, copy_original_image)
    cv2.waitKey()

    # if this is the first contour we are examining (the left-most contour), we presume this is the reference object
    if reference_object is None:
        # unpack the ordered bounding box
        (tl, tr, br, bl) = box_points
        # compute the midpoint between the top-left and bottom-left coordinates
        (tl_bl_X, tl_bl_Y) = midpoint(tl, bl)
        # the midpoint between the top-right and bottom-right
        (tr_br_X, tr_br_Y) = midpoint(tr, br)

        # compute the Euclidean distance between the midpoints
        distance_edge_by_pixel = distance.euclidean((tl_bl_X, tl_bl_Y), (tr_br_X, tr_br_Y))

        # caculate pixels Per Metric
        pixelsPerMetric = distance_edge_by_pixel / reference_object_width_by_metric
        print("pixelsPerMetric = distance_edge_by_pixel / reference_object_width_by_metric = {} / {} = {}".format(
            distance_edge_by_pixel, reference_object_width_by_metric, pixelsPerMetric))

        # construct the reference object
        reference_object = (rect, (center_X, center_Y), pixelsPerMetric)
        continue

    cv2.drawContours(copy_original_image, [reference_object[0].astype("int")], -1, (0, 255, 0), 2)
    # show reference object
    window_name = 'Reference Object'
    cv2.imshow(window_name, copy_original_image)
    cv2.waitKey()

    # draw the outline of the rotated bounding box
    cv2.drawContours(copy_original_image, [rect.astype("int")], -1, (0, 255, 0), 2)
    # show target object
    window_name = 'Target Object'
    cv2.imshow(window_name, copy_original_image)
    cv2.waitKey()

    # stack the coordinates of reference object and the coordinates of target object  to include the center of object
    reference_coordinates = np.vstack([reference_object[0], reference_object[1]])
    target_coordinates = np.vstack([rect, (center_X, center_Y)])

    colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))
    # loop over the original points
    for ((reference_X, reference_Y), (target_X, target_Y), color) in zip(reference_coordinates, target_coordinates,
                                                                         colors):
        # draw circles corresponding to the current points and connect them with a line
        cv2.circle(copy_original_image, (int(reference_X), int(reference_Y)), 5, color, -1)
        cv2.circle(copy_original_image, (int(target_X), int(target_Y)), 5, color, -1)
        cv2.line(copy_original_image, (int(reference_X), int(reference_Y)), (int(target_X), int(target_Y)), color, 2)

        # show connect reference object and target object
        window_name = 'Reference object and target object connection'
        cv2.imshow(window_name, copy_original_image)
        cv2.waitKey()

        # compute the Euclidean distance between the coordinates
        distance_between_reference_to_target_object_by_pixel = distance.euclidean((reference_X, reference_Y),
                                                                                  (target_X, target_Y))

        # convert the distance in pixels to distance in metric
        distance_between_reference_to_target_object_by_metric = \
            distance_between_reference_to_target_object_by_pixel / reference_object[2]

        # draw distance text
        (mid_X, mid_Y) = midpoint((reference_X, reference_Y), (target_X, target_Y))
        cv2.putText(copy_original_image, "{:.1f}in".format(distance_between_reference_to_target_object_by_metric),
                    (int(mid_X), int(mid_Y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # show the output image
        window_name = 'Estimating Distance'
        cv2.imshow(window_name, copy_original_image)
        cv2.waitKey()
