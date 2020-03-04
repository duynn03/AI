import numpy as np


# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

#  Malisiewicz et al.
# List of bounding boxes (startX, startY, endX, endY)
# overlap threshold
def non_max_suppression_fast(boundingBoxes, overlapThresh):
    # if there are no boundingBoxes, return an empty list
    if len(boundingBoxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats
    if boundingBoxes.dtype.kind == "i":
        boundingBoxes = boundingBoxes.astype("float")

    # initialize the list of keeping bounding boxes (indexes)
    # (the bounding boxes that we would like to keep, discarding the rest)
    keeping_boundingBox_indexs = []

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
        keeping_boundingBox_indexs.append(keeping_boudingBox_index)

        # find the largest (x, y) coordinates for the start of the bounding box
        max_start_X = np.maximum(start_X[keeping_boudingBox_index], start_X[sorted_indexs[:len(sorted_indexs) - 1]])
        max_start_Y = np.maximum(start_Y[keeping_boudingBox_index], start_Y[sorted_indexs[:len(sorted_indexs) - 1]])

        # find the smallest (x, y) coordinates for the end of the bounding box
        min_end_X = np.minimum(end_X[keeping_boudingBox_index], end_X[sorted_indexs[:len(sorted_indexs) - 1]])
        min_end_Y = np.minimum(end_Y[keeping_boudingBox_index], end_Y[sorted_indexs[:len(sorted_indexs) - 1]])

        # compute the width and height of the bounding box overlap
        width_overlap = np.maximum(0, min_end_X - max_start_X + 1)
        height_overlap = np.maximum(0, min_end_Y - max_start_Y + 1)

        # compute the ratio of overlap between the computed bounding box and the bounding box in the area list
        overlap_ratio = (width_overlap * height_overlap) / areas[sorted_indexs[:len(sorted_indexs) - 1]]

        # delete all indexes from the index list that have
        sorted_indexs = np.delete(sorted_indexs, np.concatenate(
            ([len(sorted_indexs) - 1], np.where(overlap_ratio > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the integer data type
    return boundingBoxes[keeping_boundingBox_indexs].astype("int")
