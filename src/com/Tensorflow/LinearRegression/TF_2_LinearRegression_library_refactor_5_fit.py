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
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# define optimizer & loss function
model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate)
)

# reshape data về dạng (N,1) với N là batch size
x_train = x_train.reshape(-1, 1)

# training phase
history = model.fit(x_train, y_train, epochs=epochs, verbose=False)

# visualize loss
plt.plot(history.history['loss'])
plt.show()

# visualize result
print("w: ", model.variables[0])
print("b: ", model.variables[1])
plt.plot(x_train, model(x_train))
plt.plot(x_train, y_train, '.')
plt.show()
