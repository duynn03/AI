# coding=utf-8

import sys

import imageio
import numpy as np
from sklearn import linear_model  # for logistic regression
from sklearn.metrics import accuracy_score  # for evaluation

# https://machinelearningcoban.com/2017/02/11/binaryclassifiers/#-bai-toan-phan-biet-gioi-tinh-dua-tren-anh-khuon-mat

np.random.seed(1)  # for fixing random values

# Lấy ảnh của 25 nam và 25 nữ đầu tiên làm tập training set; và 25 nam và 25 nữ còn lại làm test set.
# Mỗi bức ảnh trong AR Face thu gọn được đặt tên dưới dạng G-xxx-yy.bmp
# Trong đó:
#       G nhận một trong hai giá trị M (man) hoặc W (woman)
#       xxx là id của người, nhận gía trị từ 001 đến 050
#       yy là điều kiện chụp, nhận giá trị từ 01 đến 26,
#       trong đó
#           các điều kiện có số thứ tự từ 01 đến 07 và từ 14 đến 20 là các khuôn mặt không bị che bởi kính hoặc khăn.
#           tạm gọi mỗi điều kiện là một view.


# Phân chia training set và test set, lựa chọn các views.
path = sys.path[1] + "\\resources\\dataset\\AR\\"  # path to the database
train_ids = np.arange(1, 26)
test_ids = np.arange(26, 50)
view_ids = np.hstack((np.arange(1, 8), np.arange(14, 21)))


# Feature Extraction
# mỗi bức ảnh có kích thước 3x165x120 (số channels 3 (3 màu: red, green, blue), chiều cao 165, chiều rộng 120) là một số khá lớn nên ta sẽ làm thực hiện Feature Extraction bằng hai bước đơn giản sau
#       Chuyển ảnh màu về ảnh xám theo công thức Y' = 0.299 R + 0.587 G + 0.114 B  (reference https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems)
#       Kéo dài ảnh xám thu được thành 1 vector hàng có số chiều 165x120, sau đó sử dụng một random projection matrix để giảm số chiều về 500 (có thể thay giá trị này bằng các số khác nhỏ hơn 1000)

def rgb2gray(rgb):
    # Y' = 0.299 R + 0.587 G + 0.114 B
    return rgb[:, :, 0] * .299 + rgb[:, :, 1] * .587 + rgb[:, :, 2] * .114


# Tạo random projection matrix.
D = 165 * 120  # original dimension
d = 500  # new dimension
# generate the projection matrix
ProjectionMatrix = np.random.randn(D, d)


# feature extraction
def vectorize_img(filename):
    # load image
    rgb = imageio.imread(filename)
    # convert to gray scale
    gray = rgb2gray(rgb)
    # vectorization each row is a datas point
    im_vec = gray.reshape(1, D)
    return im_vec


# Xây dựng danh sách các tên files
# INPUT:
#     pre = 'M-' or 'W-'
#     img_ids: indexes of images
#     view_ids: indexes of views
# OUTPUT:
#     a list of filenames
def build_list_fn(pre, img_ids, view_ids):
    list_fn = []
    for im_id in img_ids:
        for v_id in view_ids:
            fn = path + pre + str(im_id).zfill(3) + '-' + str(v_id).zfill(2) + '.bmp'
            list_fn.append(fn)
    return list_fn


def build_data_matrix(img_ids, view_ids):
    total_imgs = img_ids.shape[0] * view_ids.shape[0] * 2

    X_full = np.zeros((total_imgs, D))
    y = np.hstack((np.zeros((int(total_imgs / 2),)), np.ones((int(total_imgs / 2),))))

    list_fn_m = build_list_fn('M-', img_ids, view_ids)
    list_fn_w = build_list_fn('W-', img_ids, view_ids)
    list_fn = list_fn_m + list_fn_w

    for i in range(len(list_fn)):
        X_full[i, :] = vectorize_img(list_fn[i])

    X = np.dot(X_full, ProjectionMatrix)
    return (X, y)


# X_train_full, X_test_full là các ma trận dữ liệu đã được giảm số chiều nhưng chưa được chuẩn hóa.
(X_train_full, y_train) = build_data_matrix(train_ids, view_ids)

# chuẩn hóa dữ liệu dựa vào x_mean và x_var của X_train_full
# phương pháp chuẩn hóa dữ liệu Standardization (https://machinelearningcoban.com/general/2017/02/06/featureengineering/#standardization)
# x_mean là vector kỳ vọng của toàn bộ dữ liệu training
x_mean = X_train_full.mean(axis=0)
# x_var là vector phương sai của toàn bộ dữ liệu training
x_var = X_train_full.var(axis=0)


def feature_extraction(X):
    return (X - x_mean) / x_var


X_train = feature_extraction(X_train_full)
X_train_full = None  ## free this variable

(X_test_full, y_test) = build_data_matrix(test_ids, view_ids)
X_test = feature_extraction(X_test_full)
X_test_full = None

# thực hiện thuật toán LogisticRegression
# dự đoán output của test datas và đánh giá kết quả
logreg = linear_model.LogisticRegression(C=1e5)  # just a big number
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print("Accuracy: %.2f %%" % (100 * accuracy_score(y_test, y_pred)))
