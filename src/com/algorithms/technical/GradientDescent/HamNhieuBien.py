# coding=utf-8
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(2)

# tạo 1000 điểm dữ liệu được chọn gần với đường thẳng y = 4 + 3x
X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added

# TÌM NGHIỆM THEO CÔNG THỨC
# Building Xbar
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b)
print('Solution found by formula: w = ',w_lr.T)

# Display result
w = w_lr
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1*x0

# Draw the fitting line
# Đường thẳng tìm được là đường có màu vàng có phương trình y = 4.0071715 + 2.98225924x
plt.plot(X.T, y.T, 'b.')     # data
plt.plot(x0, y0, 'y', linewidth = 2)   # the fitting line
plt.axis([0, 1, 0, 10])
plt.show()

# tính loss function
def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2

# tính đạo hàm (ước lượng) của loss function
def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

# Numerical gradient (kiểm tra đạo hàm có đúng ko)
# parameter
#           w: kết quả ma trận w tìm ra (thông qua ước lượng đạo hàm)
#           cost: loss function
#           grad: hàm ước lượng đạo hàm
# return về true nếu sai số ước lượng đạo hàm < e^(-10^6)
def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

def numerical_grad(w, cost):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps
        w_n[i] -= eps
        g[i] = (cost(w_p) - cost(w_n))/(2*eps)
    return g
print( 'Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))

# TÌM NGHIỆM THEO GRADIENT DESENT
# main Gradient Desent (nhiều chiều):
# parameter
#           eta: learning rate
#           grad: đạo hàm
def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad(w[-1])
        w.append(w_new)
        if has_converged(w_new, grad):
            break
    return (w, it)

# check convergence
# Nếu norm của gradient quá nhỏ thì ta sẽ dừng lại
def has_converged(w_new, grad):
    return np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3

w_init = np.array([[2], [1]])
(w1, it1) = myGD(w_init, grad, 1)
# demo: https://machinelearningcoban.com/assets/GD/img1_1.gif
print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' %(it1+1))


