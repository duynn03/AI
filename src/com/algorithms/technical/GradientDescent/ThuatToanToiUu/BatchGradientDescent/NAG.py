# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# https://machinelearningcoban.com/2017/01/16/gradientdescent2/#mot-vi-du-nho
# https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/GD/LR%20NAG.ipynb

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

# TÌM NGHIỆM THEO NAG
# parameter:
#           theta_init: điểm đặt bi ban đầu
#           grad(theta): là đạo hàm của loss function tại điểm theta
#           gamma: vận tốc trước đó (thường chọn là 0.9)
#           eta: learning rate
def GD_NAG(theta_init, grad, eta, gamma):
    theta = [theta_init]
    v_old = [np.zeros_like(theta_init)]
    for it in range(100):
        v_new = gamma*v_old[-1] + eta*grad(theta[-1] - gamma*v_old[-1])
        theta_new = theta[-1] - v_new
        theta.append(theta_new)
        v_old.append(v_new)
        if has_converged(theta_new, grad):
            break
    # this variable includes all points in the path
    # if you just want the final answer, use `return theta[-1]`
    return (theta, it)

# check convergence
# Nếu norm của gradient quá nhỏ thì ta sẽ dừng lại
def has_converged(theta_new, grad):
    return np.linalg.norm(grad(theta_new)) / len(theta_new) < 1e-3

w_init = np.array([[2], [1]])
(w_mm, it_mm) = GD_NAG(w_init, grad, .5, 0.9)
# demo: https://machinelearningcoban.com/assets/GD/LR_NAG_contours.gif
print(it_mm, w_mm)

# LR NAG with contours
N = X.shape[0]
a1 = np.linalg.norm(y, 2)**2/N
b1 = 2*np.sum(X)/N
c1 = np.linalg.norm(X, 2)**2/N
d1 = -2*np.sum(y)/N
e1 = -2*X.T.dot(y)/N

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
xg = np.arange(1.5, 7.0, delta)
yg = np.arange(0.5, 4.5, delta)
Xg, Yg = np.meshgrid(xg, yg)
Z = a1 + Xg**2 +b1*Xg*Yg + c1*Yg**2 + d1*Xg + e1*Yg


def save_gif2(eta, gamma):
    (w, it) = GD_NAG(w_init, grad, eta, gamma)
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.cla()
    plt.axis([1.5, 7, 0.5, 4.5])

    #     x0 = np.linspace(0, 1, 2, endpoint=True)

    def update(ii):
        if ii == 0:
            plt.cla()
            CS = plt.contour(Xg, Yg, Z, 100)
            manual_locations = [(4.5, 3.5), (4.2, 3), (4.3, 3.3)]
            animlist = plt.clabel(CS, inline=.1, fontsize=10, manual=manual_locations)
            #             animlist = plt.title('labels at selected locations')
            plt.plot(w_exact[0], w_exact[1], 'go')
        else:
            animlist = plt.plot([w[ii - 1][0], w[ii][0]], [w[ii - 1][1], w[ii][1]], 'r-')
        animlist = plt.plot(w[ii][0], w[ii][1], 'ro', markersize=4)
        xlabel = '$\eta =$ ' + str(eta) + '; iter = %d/%d' % (ii, it)
        xlabel += '; ||grad||_2 = %.3f' % np.linalg.norm(grad(w[ii]))
        ax.set_xlabel(xlabel)
        return animlist, ax

    anim1 = FuncAnimation(fig, update, frames=np.arange(0, it), interval=200)
    #     fn = 'img2_' + str(eta) + '.gif'
    fn = 'LR_NAG_contours.gif'
    anim1.save(fn, dpi=100, writer='imagemagick')


eta = 1
gamma = .9
save_gif2(eta, gamma)
# save_gif2(.1)
# save_gif2(2)