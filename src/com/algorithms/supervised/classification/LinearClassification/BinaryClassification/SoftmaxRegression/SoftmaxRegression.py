# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

np.random.seed(1)  # for fixing random values


# https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/13_softmax/Softmax%20Regression.ipynb

# parameter:
#       Z(scores): ma trận với mỗi cột là 1 vector z
#       output: ma trận với mỗi cột là giá trị a = softmax_stable(z)
# VD: https://machinelearningcoban.com/assets/13_softmax/softmax_ex.png
def softmax_stable(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = e_Z / e_Z.sum(axis=0)
    return A


# tạo simulated data (randomly generate data)
# số điểm dữ liệu chỉ là N = 2, số chiều dữ liệu d = 2, và số classes C = 3
N = 2  # number of training sample
d = 2  # data dimension
C = 3  # number of classes

X = np.random.randn(d, N)
y = np.random.randint(0, 3, (N,))


# One-hot coding
# convert 1d label to a matrix label
# reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
def convert_labels(y, C=C):
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape=(C, len(y))).toarray()
    return Y


Y = convert_labels(y, C)


# check đạo hàm của loss function đúng ko
# tính loss function
def cost(X, Y, W):
    A = softmax_stable(W.T.dot(X))
    return -np.sum(Y * np.log(A))


# tính đạo hàm (ước lượng) của loss function
def grad(X, Y, W):
    A = softmax_stable((W.T.dot(X)))
    E = A - Y
    return X.dot(E.T)


def numerical_grad(X, Y, W, cost):
    eps = 1e-6
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_p = W.copy()
            W_n = W.copy()
            W_p[i, j] += eps
            W_n[i, j] -= eps
            g[i, j] = (cost(X, Y, W_p) - cost(X, Y, W_n)) / (2 * eps)
    return g


# Numerical gradient (kiểm tra đạo hàm có đúng ko)
# parameter
#           w: kết quả ma trận w tìm ra (thông qua ước lượng đạo hàm)
#           cost: loss function
#           grad: hàm ước lượng đạo hàm
# return về true nếu sai số ước lượng đạo hàm < e^(-10^6)
def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(X, Y, w)
    grad2 = numerical_grad(X, Y, w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False


W_init = np.random.randn(d, C)
print('Checking gradient...', check_grad(W_init, cost, grad))


# hàm chính theo SGD
# parameter:
#   output: ma trận W cần tìm
def softmax_regression(X, y, W_init, eta, tol=1e-4, max_count=10000):
    W = [W_init]
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    N = X.shape[1]
    d = X.shape[0]

    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            ai = softmax_stable(np.dot(W[-1].T, xi))
            W_new = W[-1] + eta * xi.dot((yi - ai).T)
            count += 1
            # stopping criteria
            if count % check_w_after == 0:
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    return W
            W.append(W_new)
    return W


eta = .05
d = X.shape[0]
W_init = np.random.randn(d, C)

W = softmax_regression(X, y, W_init, eta)
# W[-1] is the solution, W is all history of weights
print(W[-1])


# class của 1 new data có thể tìm bằng cách xác định vị trí của giá trị lớn nhất ở đầu ra dự đoán (tương ứng với xác suất điểm dữ liệu rơi vào class đó là lớn nhất)
# các class được đánh số là 0, 1, 2, ..., C.
def pred(W, X):
    A = softmax_stable(W.T.dot(X))
    return np.argmax(A, axis=0)


means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

# each column is a datapoint
X = np.concatenate((X0, X1, X2), axis=0).T
# extended data
X = np.concatenate((np.ones((1, 3 * N)), X), axis=0)
C = 3

original_label = np.asarray([0] * N + [1] * N + [2] * N).T


def display(X, label):
    # K = np.amax(label) + 1
    X0 = X[:, label == 0]
    X1 = X[:, label == 1]
    X2 = X[:, label == 2]

    plt.plot(X0[0, :], X0[1, :], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[0, :], X1[1, :], 'go', markersize=4, alpha=.8)
    plt.plot(X2[0, :], X2[1, :], 'rs', markersize=4, alpha=.8)

    # plt.axis('equal')
    plt.axis('off')
    plt.plot()
    plt.show()


display(X[1:, :], original_label)

W_init = np.random.randn(X.shape[0], C)
W = softmax_regression(X, original_label, W_init, eta)
print(W[-1])

# Visualize
xm = np.arange(-2, 11, 0.025)
xlen = len(xm)
ym = np.arange(-3, 10, 0.025)
ylen = len(ym)
xx, yy = np.meshgrid(xm, ym)

xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)

XX = np.concatenate((np.ones((1, xx.size)), xx1, yy1), axis=0)

# Dự đoán kết quả
Z = pred(W[-1], XX)
print(Z)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
CS = plt.contourf(xx, yy, Z, 200, cmap='jet', alpha=.1)
plt.xlim(-2, 11)
plt.ylim(-3, 10)
plt.xticks(())
plt.yticks(())
display(X[1:, :], original_label)
plt.show()
