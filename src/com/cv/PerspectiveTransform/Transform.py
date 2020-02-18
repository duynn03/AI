import cv2
import numpy as np


# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

def order_points(points):
    # initialzie a list of coordinates that will be ordered such that
    #       the first entry in the list is the top-left,
    #       the second entry is the top-right
    #       the third is the bottom-right,
    #       the fourth is the bottom-left

    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum (x + y)
    # whereas the bottom-right point will have the largest sum (x + y)
    sum = points.sum(axis=1)
    rect[0] = points[np.argmin(sum)]
    rect[2] = points[np.argmax(sum)]

    # now, compute the difference between the points:
    #       the top-right point will have the smallest difference (x – y)
    #       whereas the bottom-left will have the largest difference (x – y)
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    # return the ordered coordinates
    return rect

# points là 4 điểm point bao quanh ROI image
def four_point_transform(image, points):
    # obtain a consistent order of the points and unpack them individually
    rect = order_points(points)

    (tl, tr, br, bl) = rect
    # compute the width of the new image
    #       which will be the maximum distance between bottom-right and bottom-left x-coordiates
    #       or the top-right and top-left x-coordinates
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(width_bottom), int(width_top))

    # compute the height of the new image
    #       which will be the maximum distance between the top-right and bottom-right y-coordinates
    #       or the top-left and bottom-left y-coordinates
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(height_right), int(height_left))

    # now that we have the dimensions of the new image
    # construct the set of destination points to obtain a "birds eye view",(i.e. top-down view) of the image,
    # again specifying points in the top-left, top-right, bottom-right, and bottom-left order
    destination_point = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, destination_point)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped
