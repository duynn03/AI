import sys

import cv2

# get image path
image_path = sys.path[0] + "\\image\\"

# read image

logo_image = cv2.imread(image_path + 'logo.png', cv2.IMREAD_COLOR)

# show logo
window_name = 'Logo Image'
cv2.imshow(window_name, logo_image)
cv2.waitKey()

# convert the logo to gray scale
gray_logo = cv2.cvtColor(logo_image, cv2.COLOR_BGR2GRAY)

# show gray logo
window_name = 'Gray Logo'
cv2.imshow(window_name, gray_logo)
cv2.waitKey()

# Threshold logo
threshold_logo = cv2.threshold(gray_logo, 220, 255, cv2.THRESH_BINARY_INV)[1]

# show Threshold logo
window_name = 'Threshold Logo'
cv2.imshow(window_name, threshold_logo)
cv2.waitKey()

# inverse Threshold Logo
threshold_logo_inverse = cv2.bitwise_not(threshold_logo)

# show Threshold Logo Inverse
window_name = 'Threshold Logo Inverse'
cv2.imshow(window_name, threshold_logo_inverse)
cv2.waitKey()

# original image
original_image = cv2.imread(image_path + 'coin.jpg', cv2.IMREAD_COLOR)

# show image
window_name = 'Original Image'
cv2.imshow(window_name, original_image)
cv2.waitKey()

# get space of original_image to place logo
logo_rows, logo_cols, _ = logo_image.shape
roi_image = original_image[0:logo_rows, 0:logo_cols]

# show ROI image
window_name = 'ROI Image'
cv2.imshow(window_name, roi_image)
cv2.waitKey()

# delete space of logo in roi image (merge threshold_logo_inverse to roi_image)
background_logo = cv2.bitwise_and(roi_image, roi_image, mask=threshold_logo_inverse)

# show background logo
window_name = 'Background logo '
cv2.imshow(window_name, background_logo)
cv2.waitKey()

# get foreground logo
foreground_logo = cv2.bitwise_and(logo_image, logo_image, mask=threshold_logo)

# show image
window_name = 'foreground logo'
cv2.imshow(window_name, foreground_logo)
cv2.waitKey()

# add foreground logo to background logo
output_logo_image = cv2.bitwise_or(background_logo, foreground_logo)

# show image
window_name = 'output logo'
cv2.imshow(window_name, output_logo_image)
cv2.waitKey()
