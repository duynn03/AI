import cv2
import imutils

# https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/

# load the image
from src.com.cv.CenterOfContour.center_of_contour import center_of_contour

image_path = "images/"
original_image = cv2.imread(image_path + "example.png")
# show original image
window_name = 'Original Image'
cv2.imshow(window_name, original_image)
cv2.waitKey()

# convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
# show Gray image
window_name = 'Gray Image'
cv2.imshow(window_name, gray_image)
cv2.waitKey()

# Blurring
blurring_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
# show Blurring image
window_name = 'Blurring Image'
cv2.imshow(window_name, blurring_image)
cv2.waitKey()

# threshold the image
binary_image = cv2.threshold(blurring_image, 60, 255, cv2.THRESH_BINARY)[1]
# show Binary Image
window_name = 'Binary Image'
cv2.imshow(window_name, binary_image)
cv2.waitKey()

# find contours in the binary image
contours = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# parsing the contours for various versions of OpenCV.
contours = imutils.grab_contours(contours)

for (i, contour) in enumerate(contours):
    # Compute the center of each contour
    (center_X, center_Y) = center_of_contour(contour)

    # draw the contour and center of the shape on the image
    cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 2)
    cv2.circle(original_image, (center_X, center_Y), 7, (255, 255, 255), -1)
    cv2.putText(original_image, "center", (center_X - 20, center_Y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show the image
    cv2.imshow("Center", original_image)
    cv2.waitKey()
