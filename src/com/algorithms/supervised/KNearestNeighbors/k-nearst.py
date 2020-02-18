# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#https://machinelearningcoban.com/2017/01/08/knn/

# load và hiện thị vài dữ liệu mẫu (Các class được gán nhãn là 0, 1, và 2)
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target # Các class được gán nhãn là 0, 1, và 2.
print('Number of classes: %d' % len(np.unique(iris_y)))
print('Number of datas points: %d' % len(iris_y))

# Lưu ý: 2 cột cuối của matrix mang nhiều thông tin giúp ta có thể phân biệt được nó
X0 = iris_X[iris_y == 0,:]
print('\nSamples from class 0:\n', X0[:5,:])

X1 = iris_X[iris_y == 1,:]
print('\nSamples from class 1:\n', X1[:5,:])

X2 = iris_X[iris_y == 2,:]
print('\nSamples from class 2:\n', X2[:5,:])

# tách 150 dữ liệu trong Iris flower dataset ra thành 2 phần, gọi là training set (100 điểm dữ liệu) và test set (50 điểm dữ liệu)
X_train, X_test, y_train, y_test = train_test_split(
     iris_X, iris_y, test_size=50)
print("\nTraining size: %d" %len(y_train))
print("Test size    : %d" %len(y_test))

# k=1 nghĩa là: mỗi điểm test datas, chỉ xét 1 điểm training datas gần nhất và lấy label của điểm đó để dự đoán cho điểm test này.
# p = 2 nghĩa là norm 2 (chỉ tính khoảng cách của 2 cột cuối của matrix)
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Print results for 20 test datas points:")
print("Predicted labels: ", y_pred[20:40])
print("Ground truth    : ", y_test[20:40])

# evaluation
print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
# ==> kết quả 94% khá chính xác (chỉ lệch kết quả của 2 3 điểm datas)

# nếu chỉ xét 1 điểm gần nhất có thể dẫn đến kết quả sai nếu điểm đó là nhiễu
# ==> để tăng độ chính xác thì tăng số lượng điểm lân cận lên
# => tăng lên 10 (nghĩa là class nào chiếm đa số thì dự đoán kết quả của class đó ) ==> kỹ thuật major voting
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy of 10NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))


# Đánh trọng số cho các điểm lân cận
# parameter weights = 'distance' là đánh trọng số
# default là 'uniform': nghĩa là coi tất cả các điểm lân cận có giá trị như nhau
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 5, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy of 10NN (1/distance weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))

# custom weights
def myweight(distances):
    sigma2 = .5 # we can change this number
    return np.exp(-distances**2/sigma2)

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy of 10NN (customized weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))