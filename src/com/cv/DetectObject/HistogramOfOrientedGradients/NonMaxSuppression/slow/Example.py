import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.com.cv.DetectObject.HistogramOfOrientedGradients.NonMaxSuppression.slow.non_max_suppression_slow import \
    non_max_suppression_slow

# load the image
image_path = "images/"
original_image = cv2.imread(image_path + "example_01.jpg")
# show original image
window_name = 'Original Image'
cv2.imshow(window_name, original_image)
cv2.waitKey()

# example_01
# startX, startY, endX, endY
boundingBoxes = np.array([
    (12, 84, 140, 212),
    (24, 84, 152, 212),
    (36, 84, 164, 212),
    (12, 96, 140, 224),
    (24, 96, 152, 224),
    (24, 108, 152, 236)])

# example_02
# startX, startY, endX, endY
# boundingBoxes = np.array([
#         (114, 60, 178, 124),
#         (120, 60, 184, 124),
#         (114, 66, 178, 130)])

# example_03
# startX, startY, endX, endY
# boundingBoxes = np.array([
#         (12, 30, 76, 94),
#         (12, 36, 76, 100),
#         (72, 36, 200, 164),
#         (84, 48, 212, 176)])

# clone original image
before_perform_non_max_suppression = original_image.copy()
print("Before applying non-maximum: %d bounding boxes" % (len(boundingBoxes)))
# draw bounding boxes
colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255), (255, 0, 0))
for ((startX, startY, endX, endY), color) in zip(boundingBoxes, colors):
    cv2.rectangle(before_perform_non_max_suppression, (startX, startY), (endX, endY), color, 2)
# show original image
window_name = 'Before applying non-maximum'
cv2.imshow(window_name, before_perform_non_max_suppression)
cv2.waitKey()


# clone original image
after_perform_non_max_suppression = original_image.copy()
# perform non-maximum suppression on the bounding boxes
pick = non_max_suppression_slow(original_image.copy(), colors, boundingBoxes, 0.3)
print("After applying non-maximum: %d bounding boxes" % (len(pick)))
# draw bounding boxes
for (startX, startY, endX, endY) in pick:
    cv2.rectangle(after_perform_non_max_suppression, (startX, startY), (endX, endY), (0, 255, 0), 2)
# show original image
window_name = 'After applying non-maximum'
cv2.imshow(window_name, after_perform_non_max_suppression)
cv2.waitKey()

# show Result
fig = plt.figure("Result")
images = ("Before", before_perform_non_max_suppression), \
         ("After ", after_perform_non_max_suppression)
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()
