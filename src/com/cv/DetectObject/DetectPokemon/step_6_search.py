import pickle
import sys

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

from src.com.algorithms.technical.DetectObject.Searcher import Searcher
from src.com.algorithms.technical.DetectObject.zernikemoments import ZernikeMoments

# load the index
indexs_path = sys.path[0] + "\\indexs\\"
index_file = open(indexs_path + "index.cpickle", "rb").read()
index = pickle.loads(index_file)

# load the pokemon image, convert it to grayscale, and resize it
pokemon_index = sys.path[0] + "\\"
pokemon_image = cv2.imread(pokemon_index + "pokemon_image.png")
# show pokemon image
window_name = 'Pokemon Image'
cv2.imshow(window_name, pokemon_image)
cv2.waitKey()

pokemon_gray_image = cv2.cvtColor(pokemon_image, cv2.COLOR_BGR2GRAY)
# show pokemon gray image
window_name = 'Pokemon Gray Image'
cv2.imshow(window_name, pokemon_gray_image)
cv2.waitKey()

pokemon_resized_image = imutils.resize(pokemon_gray_image, width=64)
# show pokemon Resized image
window_name = 'Pokemon Resized Image'
cv2.imshow(window_name, pokemon_resized_image)
cv2.waitKey()

# threshold the image
binary_pokemon_image = cv2.adaptiveThreshold(pokemon_resized_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 7)
# show binary pokemon image
window_name = 'Binary Pokemon Image'
cv2.imshow(window_name, binary_pokemon_image)
cv2.waitKey()

# initialize the outline image
outline_image = np.zeros(pokemon_resized_image.shape, dtype="uint8")

# find the outermost contours (the outline) of the pokemone
contours = cv2.findContours(binary_pokemon_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# parsing the contours for various versions of OpenCV.
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
# sort the contours based on their area (in descending order) and keeping only the largest contour and discard the others.

# draw contours
cv2.drawContours(outline_image, [contours], -1, 255, -1)
# show outline image
window_name = 'Outline Image'
cv2.imshow(window_name, outline_image)
cv2.waitKey()

# compute Zernike moments to characterize the shape of pokemon outline
zernikeMoments = ZernikeMoments(21)
pokemonFeatures = zernikeMoments.describe(outline_image)

# perform the search to identify the pokemon
searcher = Searcher(index)
pokemon_name = searcher.search(pokemonFeatures)
print("That pokemon is: %s" % pokemon_name[0][1].upper())

# show image Comparing
sprites_path = "sprites/"
original_image = cv2.imread(sprites_path + pokemon_name[0][1] + ".png")

fig = plt.figure("Pokemon is " + pokemon_name[0][1].upper())
images = ("Original Pokemon", original_image), ("Finded Pokemon", pokemon_image), ("Outline", outline_image)
# show the image
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()
