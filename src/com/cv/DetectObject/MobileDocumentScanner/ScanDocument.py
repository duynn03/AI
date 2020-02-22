import cv2
import imutils
import matplotlib.pyplot as plt
from skimage.filters import threshold_local

from src.com.cv.PerspectiveTransform.Transform import four_point_transform

# https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/

# load the image
image_path = "images/"
original_image = cv2.imread(image_path + "page.jpg")
# show original image
window_name = 'Original Image'
cv2.imshow(window_name, original_image)
cv2.waitKey()

# compute the ratio of the old height to the new height
ratio = original_image.shape[0] / 500.0

# clone & resized original image
resized_image = original_image.copy()
resized_image = imutils.resize(resized_image, height=500)
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
blurring_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
# show Blurring image
window_name = 'Blurring Image'
cv2.imshow(window_name, blurring_image)
cv2.waitKey()

# find edges in the image
edged_image = cv2.Canny(blurring_image, 75, 200)
# show Edged image
window_name = 'Edged Image'
cv2.imshow(window_name, edged_image)
cv2.waitKey()

# find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
contours = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# parsing the contours for various versions of OpenCV.
contours = imutils.grab_contours(contours)

# sorting our contours, from largest to smallest by calculating the area of the contour using and We now have only the 10 largest contours
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

# initialize our screen contour
screenContour = None

# loop over the contours to determine which contour is the screen contour
for c in contours:
    # approximate the contour (xấp xỉ hình dạng của contour)
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    # if our approximated contour has four points , then we can assume that we have found our screen
    # 4 điểm đại diện cho 4 điểm của hình chữ nhật cần tìm
    if len(approx) == 4:
        screenContour = approx
        break

cv2.drawContours(resized_image, [screenContour], -1, (0, 255, 0), 2)
# show Screen Contour
window_name = 'Screen Contour'
cv2.imshow(window_name, resized_image)
cv2.waitKey()

# convert points from 3D to 2D
points_2D = screenContour.reshape(4, 2)
# apply the four point transform to obtain a top-down view of the original image
warped_image = four_point_transform(original_image, points_2D * ratio)

# show warped_image
fig = plt.figure("Perspective Transform")
images = ("Original Image", imutils.resize(original_image, height=650)), (
    "Transformed", imutils.resize(warped_image, height=650))
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()

# (Do ảnh mờ nên sẽ áp dụng thresholding để làm ảnh rõ hơn)
# convert the warped image to grayscale
gray_warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
# show Gray warped image
window_name = 'Gray Transform Image'
cv2.imshow(window_name, imutils.resize(gray_warped_image, height=650))
cv2.waitKey()

# threshold it to give it that 'black and white' paper effect
T = threshold_local(gray_warped_image, 11, offset=10, method="gaussian")
binary_warped_image = (gray_warped_image > T).astype("uint8") * 255
# show Binary warped image
window_name = 'Binary Transform Image'
cv2.imshow(window_name, imutils.resize(binary_warped_image, height=650))
cv2.waitKey()

# show Result
fig = plt.figure("Result")
images = ("Original Image", imutils.resize(original_image, height=650)), (
    "Result", imutils.resize(binary_warped_image, height=650))
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()
