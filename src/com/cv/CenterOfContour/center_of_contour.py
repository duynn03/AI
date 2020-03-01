import cv2


def center_of_contour(contour):
    # compute the center (x,y) of the contour area
    M = cv2.moments(contour)
    center_X = int(M["m10"] / M["m00"])
    center_Y = int(M["m01"] / M["m00"])
    return (center_X, center_Y)
