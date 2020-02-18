import pickle
import sys

import cv2
import imutils
import numpy as np
from imutils.paths import list_images

from src.com.algorithms.technical.DetectObject.zernikemoments import ZernikeMoments

# https://stackoverflow.com/questions/36434764/permissionerror-errno-13-permission-denied

sprites_path = "sprites/"

# initialize our descriptor (Zernike Moments with a radius of 21 used to characterize the shape of our pokemon)
zernikeMoments = ZernikeMoments(21)
index = {}

# loop over the sprite images
for path in list_images(sprites_path):
    # parse out the pokemon name, then load the image and convert it to grayscale

    # extract name
    pokemon_name = path[path.rfind("/") + 1:].replace(".png", "")
    # print(pokemon_name)

    # load image
    image = cv2.imread(path)
    # show image
    # window_name = 'Original Image'
    # cv2.imshow(window_name, image)
    # cv2.waitKey()

    # convert to gray image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # show gray image
    # window_name = 'Gray Image'
    # cv2.imshow(window_name, image)
    # cv2.waitKey()

    # pad the image with extra white pixels to ensure the edges of the pokemon are not up against the borders of the image
    image = cv2.copyMakeBorder(image, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    # show Added Border image
    # window_name = 'Added Border Image'
    # cv2.imshow(window_name, image)
    # cv2.waitKey()

    # invert the image and threshold it
    binary_image = cv2.bitwise_not(image)
    # show Added Border image
    # window_name = 'Binary Image'
    # cv2.imshow(window_name, binary_image)
    # cv2.waitKey()

    binary_image[binary_image > 0] = 255

    # find the outermost contours (the outline) of the pokemone
    contours = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # parsing the contours for various versions of OpenCV.
    contours = imutils.grab_contours(contours)
    # sort the contours based on their area (in descending order) and keeping only the largest contour and discard the others.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # initialize the outline image,
    outline_image = np.zeros(image.shape, dtype="uint8")

    # draw contours
    cv2.drawContours(outline_image, [contours], -1, 255, -1)

    # show Outline image
    # window_name = 'Outline Image'
    # cv2.imshow(window_name, outline_image)
    # cv2.waitKey()

    # compute Zernike moments to characterize the shape of pokemon outline, then update the index
    pokemonFeatures = zernikeMoments.describe(outline_image)
    index[pokemon_name] = pokemonFeatures

    # feature vector is of 25-dimensionality (25 dimension đại diện cho 25 điểm của contours)
    # ==> Để phác thảo pokemon thì ta cần 25 giá trị
    # print(moments)
    # print(moments.shape)

# write the index to file
indexs_path = sys.path[0] + "\\indexs\\"
f = open(indexs_path + "index.cpickle", "wb")
f.write(pickle.dumps(index))
f.close()

