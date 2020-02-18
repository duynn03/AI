# coding=utf-8
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

np.random.seed(11)

# chọn center của 3 cluster
means = [[2, 2], [8, 3], [3, 6]]
# ma trận hiệp phương sai giống nhau và là ma trận đơn vị
cov = [[1, 0], [0, 1]]
# Mỗi cluster có 500 điểm tuần theo phân phối chuẩn nhiều chiều
# mỗi điểm dữ liệu là một hàng của ma trận dữ liệu.
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)
K = 3

original_label = np.asarray([0] * N + [1] * N + [2] * N).T


# display datas
# mỗi cluster tương ứng với một màu
def kmeans_display(X, label):
    K = np.amax(label) + 1
    # chọn điểm dữ liệu của X ở các dòng mà có labels là k.
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.show()


kmeans_display(X, original_label)


# B1: Khởi tạo các centers ban đầu
def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]


# B2: Gán label mới cho các điểm khi biết các centers (Cố định M để tìm Y)
def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw datas and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis=1)


# B3: Cập nhật các centers mới dữa trên dữ liệu vừa được gán nhãn (Cố định Y để tìm M)
def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster
        Xk = X[labels == k, :]
        # take average
        centers[k, :] = np.mean(Xk, axis=0)
    return centers


# B4: kiểm tra điều kiện dừng của thuật toán
def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) ==
            set([tuple(a) for a in new_centers]))


# main function
def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)


(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])

kmeans_display(X, labels[-1])
