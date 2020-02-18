import cv2
import imutils
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim  # require scikit-image lib

# load the images
image_path = "images/"
original_image = cv2.imread(image_path + "original_03.png", cv2.IMREAD_COLOR)
modified_image = cv2.imread(image_path + "modified_03.png", cv2.IMREAD_COLOR)

# show image
fig = plt.figure("Images")
images = ("Original Image", original_image), ("Modified Image", modified_image)
# show the image
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()

# convert the images to grayscale
original_gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
modified_gray_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)

# show gray image
fig = plt.figure("Gray Images")
images = ("Original Gray Image", original_gray_image), ("Modified Gray Image", modified_gray_image)
# show the image
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()

# compute the Structural Similarity Index (SSIM) between the two images, ensuring that the difference image is returned
# The score  represents the structural similarity index between the two input images. Value = [-1, 1]
# The diff  image contains the actual image differences between the two input images that we wish to visualize. Value = [0, 1]
(score, diff) = compare_ssim(original_gray_image, modified_gray_image, full=True)
print("SSIM: {}".format(score))

# convert diff to [0,255]
diff = (diff * 255).astype("uint8")

# get threshold image
binary_diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# show the output images
fig = plt.figure("Diff & Binary Diff")
images = ("Diff", diff), ("Binary Diff", binary_diff)
# show the image
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()

# find the contours
contours = cv2.findContours(binary_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# parsing the contours for various versions of OpenCV.
contours = imutils.grab_contours(contours)

# draw rectangles around the different regions on each image
for i in contours:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    (x, y, w, h) = cv2.boundingRect(i)
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(modified_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the output images
fig = plt.figure("Output")
images = ("Original Image", original_image), ("Modified Image", modified_image)
# show the image
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()
