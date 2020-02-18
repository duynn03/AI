import imageio
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

np.random.seed(0)

# https://towardsdatascience.com/forward-propagation-in-neural-networks-simplified-math-and-code-version-bbcfef6f9250
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
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()
        self.w4 = np.random.randn()
        self.w5 = np.random.randn()
        self.w6 = np.random.randn()
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    # calculates the output value for the given input observation
    def forward_pass(self, x):
        self.x1, self.x2 = x
        self.a1 = self.w1 * self.x1 + self.w2 * self.x2 + self.b1
        self.h1 = self.sigmoid(self.a1)
        self.a2 = self.w3 * self.x1 + self.w4 * self.x2 + self.b2
        self.h2 = self.sigmoid(self.a2)
        self.a3 = self.w5 * self.h1 + self.w6 * self.h2 + self.b3
        self.h3 = self.sigmoid(self.a3)
        # forward_matrix is a 2d array để lưu lại các thông số của mỗi observation (a1, h1, a2, h2, a3, h3)
        # visualization: https://miro.medium.com/max/808/1*v_s725vl3FTGyCLmORU3HA.png
        forward_matrix = np.array([[0, 0, 0, 0, self.h3, 0, 0, 0],
                                   [0, 0, (self.w5 * self.h1), (self.w6 * self.h2), self.b3, self.a3, 0, 0],
                                   [0, 0, 0, self.h1, 0, 0, 0, self.h2],
                                   [(self.w1 * self.x1), (self.w2 * self.x2), self.b1, self.a1, (self.w3 * self.x1),
                                    (self.w4 * self.x2), self.b2, self.a2]])
        # forward_matrices là list forward_matrix (thông số của tất cả các observations)
        forward_matrices.append(forward_matrix)
        return self.h3


forward_matrices = []
ffn = FeedForwardNetwork()
for x in X_train:
    print(ffn.forward_pass(x))


# visualize
# creates a heat map to visualize the values of forward_matrix for each observation
def plot_heat_map(observation):
    fig = plt.figure(figsize=(10, 1))
    sns.heatmap(forward_matrices[observation], annot=True, cmap=color_map, vmin=-3, vmax=3)
    plt.title("Observation " + str(observation))
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


imageio.mimsave('./forward_matrix_visualization.gif',
                [plot_heat_map(i) for i in range(0, len(forward_matrices), len(forward_matrices) // 15)], fps=1)
