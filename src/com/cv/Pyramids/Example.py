import cv2
from skimage.transform import pyramid_gaussian

from src.com.cv.Pyramids.Pyramid import pyramid

# load the image
image_path = "images/"
original_image = cv2.imread(image_path + "image.jpg")
# show original image
window_name = 'Original Image'
cv2.imshow(window_name, original_image)
cv2.waitKey()

scale = 3
# METHOD #1: No smooth, just scaling.
# (does not smooth the image with a Gaussian at each layer of the pyramid, thường sử dụng với HOG descriptor for object classification)
# loop over the image pyramid
for (i, resized) in enumerate(pyramid(original_image, scale=scale)):
    # show the resized image
    window_name = "Method #1: Layer {}".format(i + 1)
    cv2.imshow(window_name, resized)
    cv2.waitKey()

# METHOD #2: Resizing + Gaussian smoothing.
# (apply Gaussian smoothing at each layer of the pyramid, thường sử dụng với SIFT or the Difference of Gaussian keypoint detector)
minSize = (30, 30)
for (i, resized) in enumerate(pyramid_gaussian(original_image, downscale=scale)):
    # if the image is too small, break from the loop
    if resized.shape[0] < minSize[1] or resized.shape[1] < minSize[0]:
        break

    # show the resized image
    window_name = "Method #2: Layer {}".format(i + 1)
    cv2.imshow(window_name, resized)
    cv2.waitKey()
