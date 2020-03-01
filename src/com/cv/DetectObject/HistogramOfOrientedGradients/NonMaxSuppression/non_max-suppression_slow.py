import numpy as np

# https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/

#  Felzenszwalb et al.
# set of bounding boxes in the form of (startX, startY, endX, endY)
# overlap threshold
def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked bounding boxes (indexes)
    # (the bounding boxes that we would like to keep, discarding the rest)
    pick = []

    # grab the coordinates of the bounding boxes
    start_X = boxes[:, 0]
    start_Y = boxes[:, 1]
    end_X = boxes[:, 2]
    end_Y = boxes[:, 3]

    # compute the area of the bounding boxes (tính diện tích)
    area = (end_X - start_X + 1) * (end_Y - start_Y + 1)
    # sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    sorted_indexs = np.argsort(end_Y)

    # keep looping while some index still remain in the indexs list
    while len(sorted_indexs) > 0:
        # grab the last index in the indexes list
        last_index = len(sorted_indexs) - 1
        # grab the value of last element
        last_element = sorted_indexs[last_index]
        # add last element to the list of picked indexes
        pick.append(last_element)
        # initialize the suppression list (i.e. indexes that will be deleted) using the last index
        suppress = [last_index]