import cv2
import matplotlib.pyplot as plt
import numpy as np

# load the image
image_path = "images/"
original_image = cv2.imread(image_path + "image_03.png")
# show original image
window_name = 'Original Image'
cv2.imshow(window_name, original_image)
cv2.waitKey()

# clone original image
copy_original_image = original_image.copy()

# convert the image to grayscale
gray_image = cv2.cvtColor(copy_original_image, cv2.COLOR_BGR2GRAY)
# show Gray image
window_name = 'Gray Image'
cv2.imshow(window_name, gray_image)
cv2.waitKey()

# detect circles in the image
circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1.2, 75)

# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for i, (x, y, r) in enumerate(circles):
        # draw the circle in the output image
        cv2.circle(copy_original_image, (x, y), r, (0, 255, 0), 4)

        # draw a rectangle corresponding to the center of the circle
        cv2.rectangle(copy_original_image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show image
        window_name = 'Circle ' + str(i) + ' Image'
        cv2.imshow(window_name, copy_original_image)
        cv2.waitKey()

    # show Output image
    window_name = 'Circle Detecting Image'
    cv2.imshow(window_name, copy_original_image)
    cv2.waitKey()

    # show compare marking
    fig = plt.figure("Circle Detecting")
    images = ("Orignal Image", original_image), ("Circle Detecting Image", copy_original_image)
    # show the image
    for (i, (name, image)) in enumerate(images):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.set_title(name)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
    # show the figure
    plt.show()
