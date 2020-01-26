# coding=utf-8
# %reset
import numpy as np
from mnist import MNIST
from sklearn import linear_model
from sklearn.metrics import accuracy_score

# https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/13_softmax/Softmax%20Regression.ipynb

mntrain = MNIST(
    'C:/Users/Admin/Desktop/Working/Gitlap/Python/DemoMachineLearning/src/com/data/MNIST/')
mntrain.load_training()
Xtrain = np.asarray(mntrain.train_images) / 255.0
ytrain = np.array(mntrain.train_labels.tolist())

mntest = MNIST(
    'C:/Users/Admin/Desktop/Working/Gitlap/Python/DemoMachineLearning/src/com/data/MNIST/')
mntest.load_testing()
Xtest = np.asarray(mntest.test_images) / 255.0
ytest = np.array(mntest.test_labels.tolist())

# train
# solver = 'lbfgs' là một phương pháp tối ưu cũng dựa trên gradient nhưng hiệu quả hơn và phức tạp hơn Gradient Descent (Reference: https://en.wikipedia.org/wiki/Limited-memory_BFGS)
logreg = linear_model.LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
logreg.fit(Xtrain, ytrain)

# test
y_pred = logreg.predict(Xtest)
print("Accuracy: %.2f %%" % (100 * accuracy_score(ytest, y_pred.tolist())))
