# coding=utf-8
# %reset
import sys

import matplotlib.pyplot as plt
import numpy as np
from display_network import *
from mnist import MNIST  # require: pip install python-mnist
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/kmeans/Kmeans2.ipynb

# path to your MNIST folder
MNIST_path = sys.path[1] + "\\resources\\dataset\\MNIST\\"
mndata = MNIST(MNIST_path)
mndata.load_testing()
# mỗi hàng của các ma trận chứa một điểm dữ liệu (là một bức ảnh đã được vector hóa)
X = mndata.test_images
X0 = np.asarray(X)[:1000, :] / 256.0
X = X0

K = 10
kmeans = KMeans(n_clusters=K).fit(X)
pred_label = kmeans.predict(X)

print(type(kmeans.cluster_centers_.T))
print(kmeans.cluster_centers_.T.shape)
A = display_network(kmeans.cluster_centers_.T, K, 1)

f1 = plt.imshow(A, interpolation='nearest', cmap="jet")
f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)
plt.show()

# a colormap and a normalization instance
cmap = plt.cm.jet
norm = plt.Normalize(vmin=A.min(), vmax=A.max())

# map the normalized datas to colors
# image is now RGBA (512x512x4)
image = cmap(norm(A))

# scipy.misc.imsave('aa.png', image)

# Chọn một vài ảnh từ mỗi cluster.
print(type(pred_label))
print(pred_label.shape)
print(type(X0))

N0 = 20;
X1 = np.zeros((N0 * K, 784))
X2 = np.zeros((N0 * K, 784))

for k in range(K):
    Xk = X0[pred_label == k, :]

    center_k = [kmeans.cluster_centers_[k]]
    neigh = NearestNeighbors(N0).fit(Xk)
    dist, nearest_id = neigh.kneighbors(center_k, N0)

    X1[N0 * k: N0 * k + N0, :] = Xk[nearest_id, :]
    X2[N0 * k: N0 * k + N0, :] = Xk[:N0, :]

plt.axis('off')
A = display_network(X2.T, K, N0)
f2 = plt.imshow(A, interpolation='nearest')
plt.gray()
plt.show()

# import scipy.misc
# scipy.misc.imsave('bb.png', A)


# plt.axis('off')
# A = display_network(X1.T, 10, N0)
# scipy.misc.imsave('cc.png', A)
# f2 = plt.imshow(A, interpolation='nearest' )
# plt.gray()

# plt.show()
