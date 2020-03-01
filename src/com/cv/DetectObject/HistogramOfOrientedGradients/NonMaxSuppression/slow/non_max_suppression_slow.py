import cv2

import numpy as np


# https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/

#  Felzenszwalb et al.
# List of bounding boxes (startX, startY, endX, endY)
# overlap threshold
def non_max_suppression_slow(imageVisualization, colors, boundingBoxes, overlapThresh):
    # if there are no boundingBoxes, return an empty list
    if len(boundingBoxes) == 0:
        return []

    # initialize the list of keeping bounding boxes (indexes)
    # (the bounding boxes that we would like to keep, discarding the rest)
    keeping_boudingBox_indexs = []

    # grab the coordinates of the bounding boxes
    start_X = boundingBoxes[:, 0]
    start_Y = boundingBoxes[:, 1]
    end_X = boundingBoxes[:, 2]
    end_Y = boundingBoxes[:, 3]

    # sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    sorted_indexs = np.argsort(end_Y)

    # compute the area of the bounding boxes (tính diện tích)
    areas = (end_X - start_X + 1) * (end_Y - start_Y + 1)

    # keep looping while some index still remain in the indexs list
    while len(sorted_indexs) > 0:
        # grab the last index in the sorted indexs list
        keeping_boudingBox_index = sorted_indexs[len(sorted_indexs) - 1]

        # add last index to the list of keeping indexes
        keeping_boudingBox_indexs.append(keeping_boudingBox_index)

        # initialize the suppression list ((the list of boxes we want to ignore)) using the last index
        suppress = [len(sorted_indexs) - 1]

        # visualize keeping_boudingBox_index
        copy_image = imageVisualization.copy()
        # draw rectangle
        rectangle = boundingBoxes[keeping_boudingBox_index]
        cv2.rectangle(copy_image, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]),
                      colors[keeping_boudingBox_index], 2)
        # show original image
        window_name = 'Keeping Bounding Box = ' + str(keeping_boudingBox_index)
        cv2.imshow(window_name, copy_image)
        cv2.waitKey()

        # loop over all indexes in the sorted indexs list
        for i in range(0, len(sorted_indexs) - 1):
            # grab the current index
            index = sorted_indexs[i]

            # visualize current index
            copy_image = imageVisualization.copy()
            # draw rectangle
            rectangle = boundingBoxes[keeping_boudingBox_index]
            cv2.rectangle(copy_image, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]),
                          colors[keeping_boudingBox_index], 2)
            # show original image
            window_name = 'Keeping Bounding Box = ' + str(keeping_boudingBox_index)
            cv2.imshow(window_name, copy_image)
            cv2.waitKey()

            # find the largest (x, y) coordinates for the start of the bounding box
            max_start_X = max(start_X[keeping_boudingBox_index], start_X[index])
            max_start_Y = max(start_Y[keeping_boudingBox_index], start_Y[index])

            # find the smallest (x, y) coordinates for the end of the bounding box
            min_end_X = min(end_X[keeping_boudingBox_index], end_X[index])
            min_end_Y = min(end_Y[keeping_boudingBox_index], end_Y[index])

            # compute the width and height of the bounding box overlap
            width_overlap = max(0, min_end_X - max_start_X + 1)
            height_overlap = max(0, min_end_Y - max_start_Y + 1)

            # compute the ratio of overlap between the computed bounding box and the bounding box in the area list
            overlap_ratio = float(width_overlap * height_overlap) / areas[index]

            # if there is sufficient overlap, suppress the current bounding box
            if overlap_ratio > overlapThresh:
                suppress.append(i)

        # delete all indexes from the index list that are in the suppression list
        sorted_indexs = np.delete(sorted_indexs, suppress)
    # return only the bounding boxes that were picked
    return boundingBoxes[keeping_boudingBox_indexs]
