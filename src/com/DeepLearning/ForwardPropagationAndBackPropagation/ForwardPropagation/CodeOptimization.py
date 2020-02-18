import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

np.random.seed(0)
# https://towardsdatascience.com/forward-propagation-in-neural-networks-simplified-math-and-code-version-bbcfef6f9250
# Giống basic nhưng thay vì quản lý các parameter w1, w2…w6, a1, a2, h1, h2 thì sẽ cho thành các matrix để tiện tính toán

# X: là vector của input datas

# W1: là vector của matrix W của layer 1
# W2: là vector của matrix W của layer 2

# B1: là vector của matrix bias của layer 1
# B2: là vector của matrix bias của layer 2

# A1: là vector của matrix trước activation của layer 1
# A2: là vector của matrix trước activation của layer 2

# H1: là vector của matrix sau activation của layer 1
# H2: là vector của matrix sau activation của layer 2

# generate datas
data, labels = make_moons(n_samples=200, noise=0.04, random_state=0)
print(data.shape, labels.shape)
color_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "green"])
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=color_map)
plt.show()

# Splitting the datas into training and testing datas
# 150 datas for training, 50 datas for testing
X_train, X_val, Y_train, Y_val = train_test_split(data, labels, stratify=labels, random_state=0)
print(X_train.shape, X_val.shape)


# feed-forward algorithm
class FeedForwardNetwork:

    def __init__(self):
        np.random.seed(0)

        # init random w
        self.W1 = np.random.randn(2, 2)
        self.W2 = np.random.randn(2, 1)
        self.B1 = np.zeros((1, 2))
        self.B2 = np.zeros((1, 1))

    def sigmoid(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    # calculates the output value for the given input observation
    def forward_pass(self, X):
        self.A1 = np.matmul(X, self.W1) + self.B1
        self.H1 = self.sigmoid(self.A1)
        self.A2 = np.matmul(self.H1, self.W2) + self.B2
        self.H2 = self.sigmoid(self.A2)
        return self.H2


ffn = FeedForwardNetwork()
print(ffn.forward_pass(X_train))
