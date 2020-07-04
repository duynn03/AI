import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
print((tf.__version__))
np.random.seed(2020)

# tutorial: https://colab.research.google.com/drive/105aBny83-eqOgka_H5pZinIQ1PsHDppI#scrollTo=YB6UlBWG2gRm

# make sample: y = 1.8 * x + 32
def make_data():
    # generate data with some noise
    n_datapoint = 100
    x = np.random.randn(n_datapoint) * 5 + 25
    noise = np.random.randn(len(x)) * 2
    w = 1.8
    b = 32
    y = w * x + b + noise
    return x, y


x_train, y_train = make_data()

# visualize data
plt.plot(x_train, y_train, '.')
plt.xlabel("Celsius")
plt.ylabel("Fahrenheit")
plt.show()

# init hyperparameter
learning_rate = 5e-5
# epochs là số lần cập nhật toàn bộ data
epochs = 300

# init w,b parameters
w = tf.Variable(0.)
b = tf.Variable(0.)


def predict(x):
    return w * x + b


def squared_error(y_true, y_pred):
    return tf.reduce_mean((y_true - y_pred) ** 2)


# calculate loss with w,b
loss = squared_error(predict(x_train), y_train)
# print loss with init w,b
print(loss.numpy())

# training phase
loss_history = []
line_history = []

for i in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = predict(x_train)  # <- Forward
        loss = squared_error(y_pred, y_train)  # <- Compute error of prediction with ground truth
    gradients = tape.gradient(loss, [w, b])  # <- Calculate gradients of loss w.r.t weights (m, b)

    # update w & b parameters
    w.assign_sub(gradients[0] * learning_rate)  # <- Update weights (one step gradient descent)
    b.assign_sub(gradients[1] * learning_rate)

    # for logging
    loss_np = loss.numpy()
    loss_history.append(loss_np)
    line_history.append([w.numpy(), b.numpy()])
    if i % 20 == 0:
        print(f"Step {i}, loss = {loss_np}")

# visualize loss
plt.plot(loss_history)
plt.show()

# visualize result
print("w: ", w)
print("b: ", b)
plt.plot(x_train, predict(x_train))
plt.plot(x_train, y_train, '.')
plt.show()

for w, b in line_history:
    plt.plot(x_train, w * x_train + b, c='green', alpha=0.6)
plt.plot(x_train, y_train, '.')
plt.show()