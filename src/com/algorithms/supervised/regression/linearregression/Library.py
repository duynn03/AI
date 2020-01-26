# coding=utf-8
import numpy as np
from sklearn import linear_model

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False)  # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

# Compare two results
print(u'Nghiệm tìm được bằng scikit-learn  : ', regr.coef_)

w = regr.coef_
# Preparing the fitting line
w_0 = w[0][0]
w_1 = w[0][1]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1 * x0

# dự đoán
y1 = w_1*155 + w_0
y2 = w_1*160 + w_0

print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )
print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )