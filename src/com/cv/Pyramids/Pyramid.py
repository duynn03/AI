import imutils


# https://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
# https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/

# scale: which controls by how much the image is resized at each layer
#       A small scale  yields more layers in the pyramid
#       And a larger scale  yields less layers.
# minSize: which is the minimum required width and height of the layer.
#       if an image in the pyramid falls below this minSize , we stop constructing the image pyramid.
# Method #1: No smooth, just scaling.
# (does not smooth the image with a Gaussian at each layer of the pyramid, thường sử dụng với HOG descriptor for object classification)
def pyramid(image, scale=2, minSize=(30, 30)):
    # yield the original image in the pyramid (the bottom layer).
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        width = int(image.shape[1] / scale)
        image = imutils.resize(image, width=width)

        # if the resized image is too small, break from the loop
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image
