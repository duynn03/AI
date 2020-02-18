# coding=utf-8
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(11)

# chọn center của 3 cluster
means = [[2, 2], [8, 3], [3, 6]]
#ma trận hiệp phương sai giống nhau và là ma trận đơn vị
cov = [[1, 0], [0, 1]]
#Mỗi cluster có 500 điểm
#mỗi điểm dữ liệu là một hàng của ma trận dữ liệu.
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X)

# display datas
#mỗi cluster tương ứng với một màu
def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.show()

kmeans_display(X, pred_label)