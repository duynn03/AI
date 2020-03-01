import cv2


def detect(contour):
    # initialize the shape_name name
    shape_name = "unidentified"

    # approximate the contour (xấp xỉ hình dạng của contour)
    perimeter = cv2.arcLength(contour, True)
    approxCurve = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

    # if the shape name is a triangle, it will have 3 vertices
    if len(approxCurve) == 3:
        shape_name = "triangle"
    # if the shape_name has 4 vertices, it is either a square or a rectangle
    elif len(approxCurve) == 4:
        # compute the bounding box of the contour and use the bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approxCurve)
        ratio = w / float(h)
        # a square will have an aspect ratio that is approximately equal to one
        # otherwise, the shape_name is a rectangle
        shape_name = "square" if 0.95 <= ratio <= 1.05 else "rectangle"
    # if the shape_name is a pentagon, it will have 5 vertices
    elif len(approxCurve) == 5:
        shape_name = "pentagon"
    # otherwise, we assume the shape_name is a circle
    else:
        shape_name = "circle"
    # return the name of the shape_name
    return shape_name
