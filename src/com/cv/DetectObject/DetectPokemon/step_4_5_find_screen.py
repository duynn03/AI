# import the necessary packages
import cv2
import imutils
import numpy as np
from skimage import exposure

query_path = "datas/screens/"

# load the query image, compute the ratio of the old height to the new height, clone it, and resize it
original_image = cv2.imread(query_path + "screen_marowak.jpg")
print("original_image's shape: ", original_image.shape)

# resize
ratio = original_image.shape[0] / 300.0
resized_image = original_image.copy()
resized_image = imutils.resize(resized_image, height=300)
# show Resized image
window_name = 'Resized Image'
cv2.imshow(window_name, resized_image)
cv2.waitKey()

# convert the image to grayscale
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
# show Resized image
window_name = 'Gray Image'
cv2.imshow(window_name, gray_image)
cv2.waitKey()

# Blurring
blurring_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
# show Blurring image
window_name = 'Blurring Image'
cv2.imshow(window_name, blurring_image)
cv2.waitKey()

# find edges in the image
edged_image = cv2.Canny(blurring_image, 30, 200)
# show Edged image
window_name = 'Edged Image'
cv2.imshow(window_name, edged_image)
cv2.waitKey()

# find contours in the edged image, keep only the largest ones, and initialize our screen contour
contours = cv2.findContours(edged_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# sorting our contours, from largest to smallest by calculating the area of the contour using and We now have only the 10 largest contours
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenContour = None

# loop over our contours to determine which contour is the screen contour
for c in contours:
    # approximate the contour (xấp xỉ hình dạng của contour)
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * perimeter, True)
    # if our approximated contour has four points , then we can assume that we have found our screen
    # 4 điểm đại diện cho 4 điểm của hình chữ nhật cần tìm
    if len(approx) == 4:
        screenContour = approx
        break

cv2.drawContours(resized_image, [screenContour], -1, (0, 255, 0), 3)
# show Screen Contour
window_name = 'Screen Contour'
cv2.imshow(window_name, resized_image)
cv2.waitKey()

# 4 điểm trong screenContours không có thứ tự top-left, top-right, bottom-right, and bottom-left
print("ScreenContour: ", screenContour)

# -- we'll start by reshaping our contour to be our finals and initializing our output rectangle in top-left, top-right, bottom-right, and bottom-left order

# convert screenContour from 3-dimensional to 2-dimensional to apply a perspective transformation
screenContour_2D = screenContour.reshape(4, 2)
print("screenContour_2D: ", screenContour_2D)

# các điểm sẽ được sắp xếp theo thứ tự top-left, top-right, bottom-right, and bottom-left
ordered_screenContour = np.zeros((4, 2), dtype="float32")
# the top-left point has the smallest sum whereas the bottom-right has the largest sum
sum = screenContour_2D.sum(axis=1)
ordered_screenContour[0] = screenContour_2D[np.argmin(sum)]
ordered_screenContour[2] = screenContour_2D[np.argmax(sum)]
# compute the difference between the points -- the top-right will have the minumum difference and the bottom-left will have the maximum difference
diff = np.diff(screenContour_2D, axis=1)
ordered_screenContour[1] = screenContour_2D[np.argmin(diff)]
ordered_screenContour[3] = screenContour_2D[np.argmax(diff)]
print("ordered_screenContour: ", ordered_screenContour)

# multiply the rectangle by the original ratio (convert screenContour về kích cỡ của original image
original_screenContour = ordered_screenContour * ratio
print("original_screenContour: ", original_screenContour)

# now that we have our rectangle of points
(tl, tr, br, bl) = original_screenContour
# computing the width of our new image
width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

# ...and now for the height of our new image
height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
# take the maximum of the width and height values to reach our final dimensions
maxWidth = max(int(width_bottom), int(width_top))
maxHeight = max(int(height_right), int(height_left))

# construct our destination points which will be used to map the screen to a top-down, "birds eye" view
destination_point = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype="float32")
# calculate the perspective PerspectiveTransform matrix and warp the perspective to grab the screen
M = cv2.getPerspectiveTransform(original_screenContour, destination_point)
warp = cv2.warpPerspective(original_image, M, (maxWidth, maxHeight))
# show warp
window_name = 'Warp'
cv2.imshow(window_name, warp)
cv2.waitKey()

# convert the warped image to grayscale a
warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
# show warp
window_name = 'Gray Warp'
cv2.imshow(window_name, warp)
cv2.waitKey()

# adjust the intensity of the pixels to have minimum and maximum values of 0 and 255, respectively
warp = exposure.rescale_intensity(warp, out_range=(0, 255))
# show Rescale Intensity Warp
window_name = 'Rescale Intensity Warp'
cv2.imshow(window_name, warp)
cv2.waitKey()

# the pokemon we want to identify will be in the top-right corner of the warped image
# cropping pokemon
(h, w) = warp.shape
(x_pokemon, y_pokemon) = (int(w * 0.4), int(h * 0.45))
pokemon_image = warp[10:y_pokemon, w - x_pokemon:w - 10]
# show Pokemon
window_name = 'Pokemon Image'
cv2.imshow(window_name, pokemon_image)
cv2.waitKey()

# save the cropped image to file
cv2.imwrite("pokemon_image.png", pokemon_image)
