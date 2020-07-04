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

# define model (auto init weight and bias)
layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])

# reshape data về dạng (N,1) với N là batch size
x_train = x_train.reshape(-1, 1)

# define optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# define loss function
loss_function = tf.losses.MeanSquaredError()
loss = lambda: loss_function(model(x_train), y_train)  # <- Compute error of prediction with ground truth

# training phase
loss_history = []
line_history = []

for i in range(epochs):
    optimizer.minimize(loss, var_list=model.variables)

    # for logging
    loss_np = loss().numpy()
    loss_history.append(loss_np)
    if i % 20 == 0:
        print(f"Step {i}, loss = {loss_np}")

# visualize loss
plt.plot(loss_history)
plt.show()

# visualize result
print("w: ", model.variables[0])
print("b: ", model.variables[1])
plt.plot(x_train, model(x_train))
plt.plot(x_train, y_train, '.')
plt.show()