import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from imutils.perspective import four_point_transform

from src.com.cv.SortContours.SortingContours import draw_text_in_center_contour
from src.com.cv.SortContours.SortingContours import sort_contours

# https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/

# define the answer key which maps the question number to the correct answer
# Question #1: B
# Question #2: D, E
# Question #3: A, D, E
# Question #4: A, D
# Question #5: B
ANSWER_KEY = {0: [1], 1: [3, 4], 2: [0, 3, 4], 3: [0, 3], 4: [1]}

# init the number of correct user answers
user_correct_answers_number = 0

# load the image
image_path = "images/"
original_image = cv2.imread(image_path + "test_05.png")
# show original image
window_name = 'Original Image'
cv2.imshow(window_name, original_image)
cv2.waitKey()

# clone original image
copy_original_image = original_image.copy()

# convert the image to grayscale
gray_image = cv2.cvtColor(copy_original_image, cv2.COLOR_BGR2GRAY)
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

# find edges in the image
edged_image = cv2.Canny(blurring_image, 75, 200)
# show Edged image
window_name = 'Edged Image'
cv2.imshow(window_name, edged_image)
cv2.waitKey()

# find the contours in the edged image, keeping only the largest ones
contours = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# parsing the contours for various versions of OpenCV.
contours = imutils.grab_contours(contours)

# if there is at least one contour which was found
if len(contours) > 0:
    # sorting our contours, from largest to smallest by calculating the area of the contour using and We now have only the 10 largest contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

# initialize our contour
screenContour = None

# loop over the contours to determine which contour is the contour
for answer_contour in contours:
    # approximate the contour (xấp xỉ hình dạng của contour)
    perimeter = cv2.arcLength(answer_contour, True)
    approx = cv2.approxPolyDP(answer_contour, 0.02 * perimeter, True)
    # if our approximated contour has four points , then we can assume that we have found our screen
    # 4 điểm đại diện cho 4 điểm của hình chữ nhật cần tìm
    if len(approx) == 4:
        screenContour = approx
        break

# draw screen contour
cv2.drawContours(copy_original_image, [screenContour], -1, (0, 255, 0), 2)
# show Screen Contour
window_name = 'Screen Contour'
cv2.imshow(window_name, copy_original_image)
cv2.waitKey()

# convert points from 3D to 2D
points_2D = screenContour.reshape(4, 2)
# apply the four point transform to obtain a top-down view of the original image
screen_image = four_point_transform(original_image, points_2D)

# show screen image
fig = plt.figure("Perspective Transform")
images = ("Original Image", original_image), ("Screen Image", screen_image)
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()

# (Do ảnh mờ nên sẽ áp dụng thresholding để làm ảnh rõ hơn)
# convert the screen Image to grayscale
gray_screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)
# show Gray Screen Image
window_name = 'Gray Screen Image'
cv2.imshow(window_name, gray_screen_image)
cv2.waitKey()

# apply Otsu's thresholding method to binarize the warped piece of paper
binary_screen_image = cv2.threshold(gray_screen_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# show Binary Screen Image
window_name = 'Binary Screen Image'
cv2.imshow(window_name, binary_screen_image)
cv2.waitKey()

# find the contours in the Binary Screen image
screenContours = cv2.findContours(binary_screen_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# parsing the contours for various versions of OpenCV.
screenContours = imutils.grab_contours(screenContours)

# initialize our answers contour
answerContours = []

# loop over the contours to determine which contour is the answers contour
for answer_contour in screenContours:
    # compute the bounding box of each the contour, then use the bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(answer_contour)
    ratio = w / float(h)

    # detect answer (nêu width = height thì là answers contour cần tìm)
    if w >= 20 and h >= 20 and ratio >= 0.9 and ratio <= 1.1:
        answerContours.append(answer_contour)

# draw answer contours
copy_screen_image = screen_image.copy()
cv2.drawContours(copy_screen_image, answerContours, -1, (0, 255, 0), 2)
# show Answer Contour
window_name = 'Answer Contours'
cv2.imshow(window_name, copy_screen_image)
cv2.waitKey()

# SORT answer contours top-to-bottom
# draw text in center contours in screen contour image
unsorted_contour_image = copy_screen_image.copy()
# loop over the unsorted contours and draw contour
for (i, answer_contour) in enumerate(answerContours):
    unsorted_contour_image = draw_text_in_center_contour(unsorted_contour_image, answer_contour, i)

# sort the answer contours top-to-bottom
method = "top-to-bottom"
answerContours, _ = sort_contours(answerContours, method)

# draw text in center contours in screen contour image
sorted_contour_image = copy_screen_image.copy()
# loop over the sorted contours and draw contour
for (i, answer_contour) in enumerate(answerContours):
    sorted_contour_image = draw_text_in_center_contour(sorted_contour_image, answer_contour, i)

# show compare contour sorting
fig = plt.figure("Sorting Answer Contours Screen by " + method)
images = ("Unsorted", unsorted_contour_image), ("Sorted", sorted_contour_image)
# show the image
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()

# init image after each marking step
marking_image = screen_image.copy()

# caculate total number of answers
total_number_of_answers = len(answerContours)
# each question has 5 possible answers)
# np.arange(0, 30, 5): [ 0  5 10 15 20 25]
# (question_number, first_index) =
#       (0, 0)
#       (1, 5)
#       (2, 10)
#       (3, 15)
#       (4, 20)
#       (5, 25)
# (question_number là số thứ tự của câu hỏi, first_index: là số thứ tự đầu tiên của câu trả lời)
for (question_number, first_answer_index) in enumerate(np.arange(0, total_number_of_answers, 5)):

    # get contours of current question
    current_question_contours = answerContours[first_answer_index:first_answer_index + 5]

    # draw current question contours
    copy_screen_image = screen_image.copy()
    cv2.drawContours(copy_screen_image, current_question_contours, -1, (0, 255, 0), 2)
    # show Current Question Contour
    window_name = 'Question ' + str((question_number + 1)) + " Contours Image: "
    cv2.imshow(window_name, copy_screen_image)
    cv2.waitKey()

    # SORT current question contours by left-to-right
    # draw text in center contours in Current Question Contour image
    unsorted_contour_image = copy_screen_image.copy()
    # loop over the unsorted contours and draw contour
    for (i, answer_contour) in enumerate(current_question_contours):
        unsorted_contour_image = draw_text_in_center_contour(unsorted_contour_image, answer_contour, i)

    # sort the current question contours by left to right
    method = "left-to-right"
    current_question_contours, _ = sort_contours(current_question_contours, method)

    # draw text in center contours in answer screen contour image
    sorted_contour_image = copy_screen_image.copy()
    # loop over the sorted contours and draw contour
    for (i, answer_contour) in enumerate(current_question_contours):
        sorted_contour_image = draw_text_in_center_contour(sorted_contour_image, answer_contour, i)

    # show compare contour sorting
    fig = plt.figure("Sorting Question " + str((question_number + 1)) + " Contours by " + method)
    images = ("Unsorted", unsorted_contour_image), ("Sorted", sorted_contour_image)
    # show the image
    for (i, (name, image)) in enumerate(images):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.set_title(name)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
    # show the figure
    plt.show()

    # initialize the index of the user answer
    user_answers = []

    # loop over the each answer contours
    for (answer_number, answer_contour) in enumerate(current_question_contours):
        # construct mask for current answer
        # Mask: định nghĩa phần nào sẽ là foreground và phần nào sẽ background sau khi merge Image
        #       vùng màu trắng - 255 sẽ là foreground
        #       phần màu đen - 0 sẽ là background
        mask_image = np.zeros(binary_screen_image.shape, dtype="uint8")
        # show Mask Image
        window_name = 'Mask Image (step 1): '
        cv2.imshow(window_name, mask_image)
        cv2.waitKey()

        # draw answer contour to mask image
        cv2.drawContours(mask_image, [answer_contour], -1, (255, 255, 255), -1)
        # show Current Mask image
        window_name = 'Mask Image (step 2): '
        cv2.imshow(window_name, mask_image)
        cv2.waitKey()

        # apply the mask to the binary screen image
        answer_contour_image = cv2.bitwise_and(binary_screen_image, binary_screen_image, mask=mask_image)
        # show Current Answer Contour Image
        window_name = 'Contour Answer ' + str((answer_number + 1)) + " of Question Image: "
        cv2.imshow(window_name, answer_contour_image)
        cv2.waitKey()

        # count the number of non-zero pixels in the answer contour area
        non_zero_pixel_amount = cv2.countNonZero(answer_contour_image)
        print("non-zero pixel amount: ", non_zero_pixel_amount)

        # nếu contour > 450 thì sẽ xác định là user chọn đáp án này
        if non_zero_pixel_amount > 650:
            answer = (answer_number, answer_contour)
            # user_answer = (non_zero_pixel_amount, answer_number)
            user_answers.append(answer)

    # initialize the index of the correct answers
    correct_answer_indexs = ANSWER_KEY[question_number]

    # init user answer index
    user_answer_indexs = []
    for user_answer in user_answers:
        user_answer_indexs.append(user_answer[0])
    print("user answer: ", user_answer_indexs)

    # check to see if the user answer is correct
    if correct_answer_indexs == user_answer_indexs:
        color = (0, 255, 0)  # green
        user_correct_answers_number += 1
    else:
        # the user answer is incorrect
        color = (0, 0, 255)  # red

    # init answer contours
    answer_contours = []
    for i in correct_answer_indexs:
        answer_contours.append(current_question_contours[i])

    # draw the outline of the correct answer on the test
    cv2.drawContours(marking_image, answer_contours, -1, color, 3)
    # show Result each Question Image
    window_name = 'Question ' + str((question_number + 1)) + ' Marking Image'
    cv2.imshow(window_name, marking_image)
    cv2.waitKey()

# init total questions number
total_question_number = 5
# caculate final user point
cv2.putText(marking_image, "{}/{}".format(user_correct_answers_number, total_question_number), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
# show Marking Image
window_name = 'Final Marking Image'
cv2.imshow(window_name, marking_image)
cv2.waitKey()

# show compare marking
fig = plt.figure("Marking")
images = ("Orignal Image", original_image), ("Marking", marking_image)
# show the image
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
# show the figure
plt.show()
