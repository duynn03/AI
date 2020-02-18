# import the necessary packages
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import compare_ssim  # require scikit-image lib


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, titleFigure):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = compare_ssim(imageA, imageB)

    # setup the figure
    fig = plt.figure(titleFigure)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()


# load the images
image_path = "images/"
# the original
original_image = cv2.imread(image_path + "original.png", cv2.IMREAD_COLOR)
# the original + contrast
contrast_image = cv2.imread(image_path + "contrast.png", cv2.IMREAD_COLOR)
# the original + photoshop
photoshopped_image = cv2.imread(image_path + "photoshopped.png", cv2.IMREAD_COLOR)

# show image
fig = plt.figure("Images")
images = ("Original", original_image), ("Contrast", contrast_image), ("Photoshopped", photoshopped_image)
# show the image
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()

# convert the images to grayscale
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
contrast_image = cv2.cvtColor(contrast_image, cv2.COLOR_BGR2GRAY)
photoshopped_image = cv2.cvtColor(photoshopped_image, cv2.COLOR_BGR2GRAY)

# initialize the figure
fig = plt.figure("Gray Images")
images = ("Gray Original", original_image), ("Gray Contrast", contrast_image), ("Gray Photoshopped", photoshopped_image)
# show the gray image
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
# show the figure
plt.show()

# compare the images
compare_images(original_image, original_image, "Original vs. Original")
compare_images(original_image, contrast_image, "Original vs. Contrast")
compare_images(original_image, photoshopped_image, "Original vs. Photoshopped")
