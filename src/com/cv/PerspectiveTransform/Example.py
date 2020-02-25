import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.com.cv.PerspectiveTransform.Transform import four_point_transform

# load the image
image_path = "images/"
original_image = cv2.imread(image_path + "example_01.png")

# grab the source coordinates (i.e. the list of of (x, y) points)
points_01 = np.array(eval("[(73, 239), (356, 117), (475, 265), (187, 443)]"), dtype="float32")
# points_02 = np.array(eval("[(101, 185), (393, 151), (479, 323), (187, 441)]"), dtype="float32")
# points_03 = np.array(eval("[(63, 242), (291, 110), (361, 252), (78, 386)]"), dtype="float32")

# apply the four point tranform
warped_image = four_point_transform(original_image, points_01)

# show the image
fig = plt.figure("Perspective Transform")
images = ("Original Image", original_image), ("Transformed", warped_image)
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()
