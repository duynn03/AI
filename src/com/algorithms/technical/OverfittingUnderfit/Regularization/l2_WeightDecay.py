# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(4)

# data: phân phối chuẩn có tâm ở [[-1, -1], [1, -1], [0, 1]].
means = [[-1, -1], [1, -1], [0, 1]]
cov = [[1, 0], [0, 1]]
N = 20
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)
K = 3

original_label = np.asarray([0] * N + [1] * N + [2] * N).T


def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=6, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=6, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=6, alpha=.8)

    plt.plot()
    # plt.show()

kmeans_display(X, original_label)
plt.show()
y = original_label.T
X = X.T