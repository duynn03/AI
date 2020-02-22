# import the necessary packages
import cv2


def sort_contours(contours, method="left-to-right"):
    # initialize the reverse flag (ascending or descending)
    #       ascending: along to the x-axis location of the bounding box of the contour
    #               ascending: left-to-right, top-to-bottom (reverse = False)
    #               descending: right-to-left or bottom-to-top (reverse = True)
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    else:
        reverse = False

    # initialize the index of the bounding box
    #       sorting by vertically - y-coordinate: top-to-bottom, bottom-to-top
    #       sorting by horizontally - x-coordinate: left-to-right, right-to-left
    if method == "top-to-bottom" or method == "top-to-bottom":
        i = 1
    else:
        i = 0

    # construct the list of bounding boxes (including x,y,w,h)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]

    # sort bounding boxes by direction (vertically or horizontally)
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (contours, boundingBoxes)

def draw_text_in_center_contour(image, contour, i):
    # compute the center (x,y) of the contour area
    M = cv2.moments(contour)
    center_X = int(M["m10"] / M["m00"])
    center_Y = int(M["m01"] / M["m00"])

    # draw a circle representing the center
    cv2.putText(image, "#{}".format(i + 1), (center_X - 25, center_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2)

    # return the image with the contour number drawn on it
    return image