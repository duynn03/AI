import sys

import cv2

# get image path
image_path = sys.path[0] + "\\image\\"

# read image
original_image = cv2.imread(image_path + 'coin.jpg', cv2.IMREAD_UNCHANGED)

# show original image
window_name = 'Original Image'
cv2.imshow(window_name, original_image)
cv2.waitKey()

# convert image to grey
grey_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# bluring to eliminate noise
blurred_image = cv2.GaussianBlur(grey_image, (15, 15), 0)

# set a thresh
thresh = 170
# get threshold image
_, binary_image = cv2.threshold(blurred_image, thresh, 255, cv2.THRESH_BINARY_INV)

# show Binary Image
window_name = 'Binary Image'
cv2.imshow(window_name, binary_image)
cv2.waitKey()

# find contours
# contours chứa các điểm ảnh của các object
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# remove non-contours parts
contours_after_remove_non_contours = []  # contours_after_remove_non_contours is pixel to draw
area_after_remove_non_contours = []  # area_after_remove_non_contours is a object frame
for i in contours:
    # area is the frame containing object (area is the number of non-zero pixels)
    area = cv2.contourArea(i)
    print(area)

    # remove contours has width & height too large (remove borders)
    (x, y, w, h) = cv2.convexityDefects()(i)
    print("width:", w)

    if area > 10000 and w < 300:
        area_after_remove_non_contours.append(area)
        contours_after_remove_non_contours.append(i)

# draw Cycle Around Contours on the empty image
image = original_image.copy()
cv2.drawContours(image, contours_after_remove_non_contours, -1, (0, 255, 0), 2)

# show Cycle Around Contours image
window_name = 'Cycle Around Contours Image'
cv2.imshow(window_name, image)
cv2.waitKey()

# draw contours index = 0
contours_0 = contours_after_remove_non_contours[0]
image = original_image.copy()
cv2.drawContours(image, contours_0, -1, (0, 255, 0), 2)

# show Cycle Around Contours image
window_name = 'Contours with index 0 Image'
cv2.imshow(window_name, image)
cv2.waitKey()

M = cv2.moments(contours_0)
print(M)
