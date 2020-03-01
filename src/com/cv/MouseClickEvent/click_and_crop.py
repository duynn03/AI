import cv2


# Anytime a mouse event happens, OpenCV will relay the pertinent details to our onMouse  function
# event:
#       The event that took place (left mouse button pressed, left mouse button released, mouse movement, etc).
# x:
#       The x-coordinate of the event.
# y:
#       The y-coordinate of the event.
# flags:
#       Any relevant flags passed by OpenCV.
# params:
#       Any extra parameters supplied by OpenCV.
def onMouse(event, x, y, flags, param):
    # grab references to the global variables
    # which is a list of two (x, y)-coordinates specifying  the rectangular region we are going to crop from our image
    global reference_points
    # if the left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # save the starting (x, y) coordinates
        reference_points = [(x, y)]

    # At this point we would drag out the rectangular region of the image that we want to crop
    # After we are done dragging out the region, we release the left mouse button
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        reference_points.append((x, y))
        # indicate that the cropping operation is finished

        # draw a rectangle around the region of interest
        cv2.rectangle(original_image, reference_points[0], reference_points[1], (0, 255, 0), 2)
        cv2.imshow(window_name, original_image)


# load the image
image_path = "images/"
original_image = cv2.imread(image_path + "example.jpg")

# create a window named "image"
window_name = "image"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, onMouse)

# clone original image
copy_original_image = original_image.copy()

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow(window_name, original_image)
    key = cv2.waitKey(1) & 0xFF
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        original_image = copy_original_image.copy()
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break
# if there are two reference points, then crop the region of interest from teh image and display it
if len(reference_points) == 2:
    roi_image = copy_original_image[reference_points[0][1]:reference_points[1][1],
                reference_points[0][0]:reference_points[1][0]]
    cv2.imshow("ROI", roi_image)
    cv2.waitKey()
