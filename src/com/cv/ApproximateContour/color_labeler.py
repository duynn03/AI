from collections import OrderedDict

import cv2
import numpy as np

# TODO: https://www.pyimagesearch.com/2016/02/15/determining-object-color-with-opencv/

class ColorLabeler:
    def __init__(self):
        # initialize the colors dictionary: containing the color name as the key and the RGB tuple as the value
        colors = OrderedDict({
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255)})
        # allocate memory for the L*a*b* image, then
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        # initialize the color names list
        self.colorNames = []

        # loop over the colors dictionary
        for (i, (color_name, rgb_value)) in enumerate(colors.items()):
            # update the L*a*b* array and the color names list
            self.lab[i] = rgb_value
            self.colorNames.append(color_name)

        # convert the L*a*b* array from the RGB color space to L*a*b*
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    # do mỗi khu vực chỉ chứa 1 color nhất định nên ta có thể tính toán được Euclidean distance giữa colors đã biết và mean image region
    def label(self, image, c):
        # construct a mask for the contour, then compute the average L*a*b* value for the masked region
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]
        # initialize the minimum distance found thus far
        minDist = (np.inf, None)
        # loop over the known L*a*b* color values
        for (i, row) in enumerate(self.lab):
            # compute the distance between the current L*a*b*
            # color value and the mean of the image
            d = dist.euclidean(row[0], mean)
            # if the distance is smaller than the current distance,
            # then update the bookkeeping variable
            if d < minDist[0]:
                minDist = (d, i)
        # return the name of the color with the smallest distance
        return self.colorNames[minDist[1]]
ColorLabeler()