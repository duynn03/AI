import time

import cv2

from src.com.cv.SlidingWindows.SlidingWindow import sliding_window
from src.com.cv.Pyramids.Pyramid import pyramid

# load the image
image_path = "images/"
original_image = cv2.imread(image_path + "image.jpg")
# show original image
window_name = 'Original Image'
cv2.imshow(window_name, original_image)
cv2.waitKey()

# define the window width and height
(window_width, window_height) = (128, 128)
# loop over the image pyramid
for (i, resized) in enumerate(pyramid(original_image, scale=1.5)):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(window_width, window_height)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != window_height or window.shape[1] != window_width:
            continue

        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE WINDOW
        # since we do not have a classifier, we'll just draw the window
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + window_width, y + window_height), (0, 255, 0), 2)
        window_name = "Layer {}".format(i + 1)
        cv2.imshow(window_name, clone)
        cv2.waitKey()
        time.sleep(0.025)
