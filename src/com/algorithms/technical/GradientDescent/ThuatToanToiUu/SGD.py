# coding=utf-8
import numpy as np

# https://machinelearningcoban.com/2017/01/16/gradientdescent2/#vi-du-voi-bai-toan-linear-regression

np.random.seed(2)

# tạo 1000 điểm dữ liệu được chọn gần với đường thẳng y = 4 + 3x
X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2*np.random.randn(1000, 1)

# Building Xbar
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_exact = np.dot(np.linalg.pinv(A), b)
print('Solution found by formula: w = ',w_exact.T)

# tính loss function
def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2;

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

# TÌM NGHIỆM THEO SGD
# parameter:
#           theta_init: điểm đặt bi ban đầu
#           grad(theta): là đạo hàm của loss function tại điểm theta
#           eta: learning rate
def SGD(theta_init, sgrad, eta):
    theta = [theta_init]
    theta_last_check = theta_init
    iter_check_theta = 10
    N = X.shape[0]
    count = 0
    for it in range(10):
        # shuffle datas
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1
            g = sgrad(theta[-1], i, rd_id)
            theta_new = theta[-1] - eta*g
            theta.append(theta_new)
            if count%iter_check_theta == 0:
                theta_this_check = theta_new
                if has_converged(theta_this_check, theta_last_check, theta_init):
                    return theta
                theta_last_check = theta_this_check
    return (theta, count)

# check convergence
# Nếu norm của gradient quá nhỏ thì ta sẽ dừng lại
def has_converged(theta_this_check, theta_last_check, theta_init):
    return np.linalg.norm(theta_this_check - theta_last_check)/len(theta_init) < 1e-3

# single point gradient
def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = Xbar[true_i, :]
    yi = y[true_i]
    a = np.dot(xi, w) - yi
    return (xi*a).reshape(2, 1)

w_init = np.array([[2], [1]])
(w_mm, it_mm) = SGD(w_init, grad, 1)
# demo: https://machinelearningcoban.com/assets/GD/LR_SGD_contours.gif
print(it_mm, w_mm)