# coding=utf-8
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt

# https://machinelearningcoban.com/2017/01/12/gradientdescent/#vi-d-n-gin-vi-python

# f(x)=x^2+5sin(x)

# tính đạo hàm
def grad(x):
    return 2*x+ 5*np.cos(x)

# để tính giá trị của hàm số
def cost(x):
    return x**2 + 5*np.sin(x)

# main Gradient Desent (1 chiều): tìm x(t+1)
# parameter
#           eta: learning rate
#           x0: điểm bắt đầu
def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        x.append(x_new)
        if abs(grad(x_new)) < 1e-3:
            break
    return (x, it)

# Điểm khởi tạo khác nhau
# x0 = -5 và x0 = 5
# demo: https://machinelearningcoban.com/assets/GD/1dimg_5_0.1_-5.gif
#       https://machinelearningcoban.com/assets/GD/1dimg_5_0.1_5.gif
(x1, it1) = myGD1(.1, -5)
(x2, it2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))

# Learning rate khác nhau
# Learning rate = 1 và Learning rate = 0.05
(x1, it1) = myGD1(5, -5)
(x2, it2) = myGD1(.05, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))