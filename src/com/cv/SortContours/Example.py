import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

from src.com.cv.SortContours.SortingContours import sort_contours
from src.com.cv.SortContours.SortingContours import draw_text_in_center_contour

# load the image
image_path = "images/"
original_image = cv2.imread(image_path + "image_01.png")
# show orginal image
window_name = 'Original Image'
cv2.imshow(window_name, original_image)
cv2.waitKey()

# initialize the accumulated edge image
accumulated_edged = np.zeros(original_image.shape[:2], dtype="uint8")

# loop over the blue, green, and red channels, respectively
for channel_image in cv2.split(original_image):
    # show channel image
    window_name = 'Channel Image'
    cv2.imshow(window_name, channel_image)
    cv2.waitKey()

    # blur the channel image to remove high frequency noise
    blurring_channel_image = cv2.medianBlur(channel_image, 11)
    # show blurring channel image
    window_name = 'Blurring Channel Image'
    cv2.imshow(window_name, blurring_channel_image)
    cv2.waitKey()

    # extract edges image
    edged_image = cv2.Canny(blurring_channel_image, 50, 200)
    # show edges image
    window_name = 'Edges Image'
    cv2.imshow(window_name, edged_image)
    cv2.waitKey()

    # update accumulate the set of edges for the image
    accumulated_edged = cv2.bitwise_or(accumulated_edged, edged_image)
    # show edges image
    window_name = 'accumulated Edge Image'
    cv2.imshow(window_name, accumulated_edged)
    cv2.waitKey()

# show the accumulated edge map
window_name = 'Final Edged Image'
cv2.imshow(window_name, accumulated_edged)
cv2.waitKey()

# find contours in the accumulated image, keeping only the largest ones
contours = cv2.findContours(accumulated_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# parsing the contours for various versions of OpenCV.
contours = imutils.grab_contours(contours)

# sorting our contours, from largest to smallest by calculating the area of the contour using and We now have only the 5 largest contours
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
# draw contours
copy_original_image = original_image.copy()
cv2.drawContours(copy_original_image, contours, -1, 255, -1)
# show Drawing contours Image
window_name = 'Drawing contours Image'
cv2.imshow(window_name, copy_original_image)
cv2.waitKey()

# draw text in center contours in origin image
unsorted_original_image = original_image.copy()
# loop over the (unsorted) contours and draw contour
for (i, contour) in enumerate(contours):
    unsorted_original_image = draw_text_in_center_contour(unsorted_original_image, contour, i)

# sort the contours according to the provided method
method = "right-to-left"
(contours, boundingBoxes) = sort_contours(contours, method)
sorted_original_image = original_image.copy()
# loop over the (now sorted) contours and draw them
for (i, contour) in enumerate(contours):
    draw_text_in_center_contour(sorted_original_image, contour, i)

# show result
fig = plt.figure("Sorting Contours by " + method)
images = ("Unsorted", unsorted_original_image), ("Sorted", sorted_original_image)
# show the image
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()
