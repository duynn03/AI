import cv2
import imutils

# https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/

# load the image
image_path = "images/"
original_image = cv2.imread(image_path + "example_02.png")
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
binary_image = cv2.threshold(blurring_image, 45, 255, cv2.THRESH_BINARY)[1]
# show Binary Image
window_name = 'Binary Image'
cv2.imshow(window_name, binary_image)
cv2.waitKey()

# perform a dilation + erosion to close gaps in between object edges
binary_image = cv2.erode(binary_image, None, iterations=2)
# show erosion image
window_name = 'Erosion'
cv2.imshow(window_name, binary_image)
cv2.waitKey()

binary_image = cv2.dilate(binary_image, None, iterations=2)
# show dilation image
window_name = 'Dilation'
cv2.imshow(window_name, binary_image)
cv2.waitKey()

# find contours in the binary image
contours = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# parsing the contours for various versions of OpenCV.
contours = imutils.grab_contours(contours)

# get hand contour (the largest one)
hand_contour = max(contours, key=cv2.contourArea)

# convert hand_contour from 3D to 2D
hand_contour = hand_contour.reshape((-1, 2))

# draw hand contour
cv2.drawContours(original_image, [hand_contour], -1, (0, 255, 255), 2)
# show hand contour image
window_name = 'Hand contour'
cv2.imshow(window_name, original_image)
cv2.waitKey()

# finds the smallest x-coordinate, left_extreme_point (west)
smallest_x_coordinate_index = hand_contour[:, 0].argmin()
left_extreme_point = tuple(hand_contour[smallest_x_coordinate_index])

# finds the largest x-coordinate, right_extreme_point (east)
largest_x_coordinate_index = hand_contour[:, 0].argmax()
right_extreme_point = tuple(hand_contour[largest_x_coordinate_index])

# finds the smallest y-coordinate, top_extreme_point (north)
smallest_y_coordinate_index = hand_contour[:, 1].argmin()
top_extreme_point = tuple(hand_contour[smallest_y_coordinate_index])

# finds the largest y-coordinate, bottom_extreme_point (south)
largest_x_coordinate_index = hand_contour[:, 1].argmax()
bottom_extreme_point = tuple(hand_contour[largest_x_coordinate_index])

#  draw each of the extreme points
# (the left-most is red, right-most is green, top-most is blue, bottom-most is teal)
cv2.circle(original_image, left_extreme_point, 8, (0, 0, 255), -1)
cv2.circle(original_image, right_extreme_point, 8, (0, 255, 0), -1)
cv2.circle(original_image, top_extreme_point, 8, (255, 0, 0), -1)
cv2.circle(original_image, bottom_extreme_point, 8, (255, 255, 0), -1)

# show the output image
cv2.imshow("Result", original_image)
cv2.waitKey()
