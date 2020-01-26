# %reset
import sys

import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST
from sklearn import linear_model
from sklearn.metrics import accuracy_score

from src.com.algorithms.unsupervised.clustering.KMeansClustering.example.ChuVietTay.display_network import \
    display_network

# https://machinelearningcoban.com/2017/02/11/binaryclassifiers/#-bai-toan-phan-biet-hai-chu-so-viet-tay


MNIST_path = sys.path[1] + "\\resources\\dataset\\MNIST\\"
mntrain = MNIST(MNIST_path)
mntrain.load_training()
# mỗi hàng của các ma trận chứa một điểm dữ liệu (là một bức ảnh đã được vector hóa)
Xtrain_all = np.asarray(mntrain.train_images)
ytrain_all = np.array(mntrain.train_labels.tolist())

mntest = MNIST(MNIST_path)
mntest.load_testing()
# mỗi hàng của các ma trận chứa một điểm dữ liệu (là một bức ảnh đã được vector hóa)
Xtest_all = np.asarray(mntest.test_images)
ytest_all = np.array(mntest.test_labels.tolist())

# lấy các hàng tương ứng với chữ số 0 và chữ số 1
#       muốn thử với cặp 3 và 4, chỉ cần thay dòng này bằng cls = [[3], [4]]
#       muốn phân loại (4, 7) và (5, 6), chỉ cần thay dòng này bằng cls = [[4, 7], [5, 6]]
#       (Các cặp bất kỳ khác đều có thể thực hiện bằng cách thay chỉ một dòng này)
cls = [[0], [1]]


# extract toàn bộ dữ liệu cho các chữ số 0 và 1 trong tập training data và test data
#  X: numpy array, matrix of size (N, d), d is data dim
#  y: numpy array, size (N, )
#  cls: two lists of labels. For example: cls = [[1, 4, 7], [5, 6, 8]]
#  return:
#      X: extracted data
#      y: extracted label
#          (0 and 1, corresponding to two lists in cls)
def extract_data(X, y, classes):
    y_res_id = np.array([])
    for i in cls[0]:
        y_res_id = np.hstack((y_res_id, np.where(y == i)[0]))
    n0 = len(y_res_id)

    for i in cls[1]:
        # chuẩn hóa để đưa dữ liệu về đoạn [0, 1] bằng cách chia toàn bộ hai ma trận dữ liệu cho 255.0
        y_res_id = np.hstack((y_res_id, np.where(y == i)[0]))
    n1 = len(y_res_id) - n0

    y_res_id = y_res_id.astype(int)
    X_res = X[y_res_id, :] / 255.0
    y_res = np.asarray([0] * n0 + [1] * n1)
    return (X_res, y_res)


# extract data for training
(X_train, y_train) = extract_data(Xtrain_all, ytrain_all, cls)

# extract data for test
(X_test, y_test) = extract_data(Xtest_all, ytest_all, cls)

# train the logistic regression model
logreg = linear_model.LogisticRegression(C=1e5)  # just a big number
logreg.fit(X_train, y_train)
# logreg.fit(Xtrain_all, ytrain_all) # trainning toàn bộ data

# predict
y_pred = logreg.predict(X_test)
# y_pred = logreg.predict(Xtest_all) # test toàn bộ data
print("Accuracy: %.2f %%" % (100 * accuracy_score(y_test, y_pred.tolist())))

# những ảnh phân loại bị sai
mis = np.where((y_pred - y_test) != 0)[0]
Xmis = X_test[mis, :]

plt.axis('off')
A = display_network(Xmis.T)
f2 = plt.imshow(A, interpolation='nearest')
plt.gray()
plt.show()
