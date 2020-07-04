import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# tutorial: https://nguyenvanhieu.vn/xay-dung-mo-hinh-linear-regression/

# Step 1: Make sample
a = [[1.31415422e-01, -2.26093368e-01],
     [-5.09640698e-01, -2.26093368e-01],
     [5.07908699e-01, -2.26093368e-01],
     [-7.43677059e-01, -1.55439190e+00],
     [1.27107075e+00, 1.10220517e+00],
     [-1.99450507e-02, 1.10220517e+00],
     [-5.93588523e-01, -2.26093368e-01],
     [-7.29685755e-01, -2.26093368e-01],
     [-7.89466782e-01, -2.26093368e-01],
     [-6.44465993e-01, -2.26093368e-01],
     [-7.71822042e-02, 1.10220517e+00],
     [-8.65999486e-04, -2.26093368e-01],
     [-1.40779041e-01, -2.26093368e-01],
     [3.15099326e+00, 2.43050370e+00],
     [-9.31923697e-01, -2.26093368e-01],
     [3.80715024e-01, 1.10220517e+00],
     [-8.65782986e-01, -1.55439190e+00],
     [-9.72625673e-01, -2.26093368e-01],
     [7.73743478e-01, 1.10220517e+00],
     [1.31050078e+00, 1.10220517e+00],
     [-2.97227261e-01, -2.26093368e-01],
     [-1.43322915e-01, -1.55439190e+00],
     [-5.04552951e-01, -2.26093368e-01],
     [-4.91995958e-02, 1.10220517e+00],
     [2.40309445e+00, -2.26093368e-01],
     [-1.14560907e+00, -2.26093368e-01],
     [-6.90255715e-01, -2.26093368e-01],
     [6.68172729e-01, -2.26093368e-01],
     [2.53521350e-01, -2.26093368e-01],
     [8.09357707e-01, -2.26093368e-01],
     [-2.05647815e-01, -1.55439190e+00],
     [-1.27280274e+00, -2.88269044e+00],
     [5.00114703e-02, 1.10220517e+00],
     [1.44532608e+00, -2.26093368e-01],
     [-2.41262044e-01, 1.10220517e+00],
     [-7.16966387e-01, -2.26093368e-01],
     [-9.68809863e-01, -2.26093368e-01],
     [1.67029651e-01, 1.10220517e+00],
     [2.81647389e+00, 1.10220517e+00],
     [2.05187753e-01, 1.10220517e+00],
     [-4.28236746e-01, -1.55439190e+00],
     [3.01854946e-01, -2.26093368e-01],
     [7.20322135e-01, 1.10220517e+00],
     [-1.01841540e+00, -2.26093368e-01],
     [-1.46104938e+00, -1.55439190e+00],
     [-1.89112638e-01, 1.10220517e+00],
     [-1.01459959e+00, -2.26093368e-01]]
X_train = np.asarray(a)
Y_train = np.asarray([399900, 329900, 369000, 232000, 539900, 299900, 314900, 198999, 212000,
                      242500, 239999, 347000, 329999, 699900, 259900, 449900, 299900, 199900,
                      499998, 599000, 252900, 255000, 242900, 259900, 573900, 249900, 464500,
                      469000, 475000, 299900, 349900, 169900, 314900, 579900, 285900, 249900,
                      229900, 345000, 549000, 287000, 368500, 329900, 314000, 299000, 179900,
                      299900, 239500])
n_samples = X_train.shape[0]

# Step 2: create graph
# Step 2.1: create placeholders for X_train and Y_train
X = tf.placeholder(tf.float32, shape=(2,), name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 2.2: create weight and bias, initialized to 0
w = tf.get_variable('weights', initializer=tf.constant([0.0, 0.0]))
b = tf.get_variable('bias', initializer=tf.constant([0.0, 0.0]))

# Step 2.3: build model to predict Y
Y_predicted = tf.multiply(tf.transpose(w), X) + b

# Step 2.4: use the squared error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')

# Step 2.5: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# Step 3: train data
# save graph
writer = tf.summary.FileWriter('/graphs', tf.get_default_graph())
with tf.Session() as session:
    # Step 3.1: initialize the necessary variables, in this case, w and b
    session.run(tf.global_variables_initializer())

    # Step 3.2: train the model for 100 epochs
    for i in range(100):
        total_loss = 0
        for x, y in zip(X_train, Y_train):
            # Session execute optimizer and fetch values of loss
            _, _loss = session.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += _loss
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    # close the writer when you're done using it
    writer.close()

    # Step 3.3: get output the values of w and b
    w_out, b_out = session.run([w, b])
    # Can also get Y_pred with sess.run
    # Y_pred = sess.run(Y_predicted, feed_dict={X: X_train, Y: Y_train})

# Step 4: predicted after train
Y_pred = X_train * w_out + b_out

# See diff between real and predict value
for i, j in zip(Y_pred, Y_train):
    print(i, '|', j)

# plot the results
plt.xlabel('x')
plt.ylabel('y')
plt.plot(X_train, Y_train, 'bo', label='Real data')
plt.plot(X_train, Y_pred, 'r', label='Predicted')
plt.legend()
plt.show()
