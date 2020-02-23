import cv2

# load the image
image_path = "images/"
original_image = cv2.imread(image_path + "cat_04.jpg")
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

# load the cat detector Haar cascade and then detect cat faces in the input image
cascades_path = "haarcascades/"
# loads Haar cascade
detector = cv2.CascadeClassifier(cascades_path + "haarcascade_frontalcatface.xml")
# detecting object
rects = detector.detectMultiScale(gray_image, scaleFactor=1.3, minSize=(75, 75))

# loop over the cat faces and draw a rectangle surrounding each
for (i, (x, y, w, h)) in enumerate(rects):
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(original_image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

# show Cat faces image
window_name = 'Cat Faces Image'
cv2.imshow(window_name, original_image)
cv2.waitKey()
