import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(2020)


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

# init hyperparameters
learning_rate = 5e-5
# epochs là số lần cập nhật toàn bộ data
epochs = 300

# init w,b parameters
w = tf.Variable(0.)
b = tf.Variable(0.)


def predict(x):
    return w * x + b

# define optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# define loss function
loss_function = tf.losses.MeanSquaredError()
loss = lambda: loss_function(predict(x_train), y_train)  # <- Compute error of prediction with ground truth

# training phase
loss_history = []
line_history = []

for i in range(epochs):
    optimizer.minimize(loss, var_list=[w, b])

    # for logging
    loss_np = loss().numpy()
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
