# The stepSize:
#       indicates how many pixels we are going to “skip” in both the (x, y) direction.
#       is determined on a per-dataset basis and is tuned to give optimal performance based on your dataset of images
#       it’s common to use a stepSize  of 4 to 8 pixels
#       the smaller your step size is, the more windows you’ll need to examine.
# The windowSize: defines the width and height of the window we are going to extract from our image (pixels)
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # returns a tuple containing the x  and y  coordinates of the sliding window, along with current window.
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
