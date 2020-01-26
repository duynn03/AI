import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Binarizer

# Assignment 1:
# Hãy viết tiếp đoạn chương trình sau đây để lọc ra số sinh viên thi không đạt( điểm < 5) môn Toán hoặc Vật Lý.
# Chỉ giữ lại số sinh viên đã thi đậu cả 2 môn trên.

students = pd.DataFrame()
students["Name"] = ["Vũ Văn Hậu", "Đào Ngọc Minh", "Trần Minh Thắng", "Hoàng Công Chất", "Lý Phương Nga"]
students["Math"] = [5, 8, 10, 4, 7]
students["Physics"] = [3, 7, 9, 6, 8]

result = students[(students["Math"] >= 5) & (students["Physics"] >= 5)]
print(result)

# Assignment 2:
# Hãy viết tiếp đoạn chương trình sau đây để thêm một cột dữ liệu “Outlier”.
# Số sinh viên thi không đạt( điểm < 5) môn Toán hoặc Vật Lý được đánh dấu “Thi lại”.
# Số sinh viên đã thi đậu cả 2 môn trên được đánh dấu “Đạt”

students["outlier"] = np.where((students["Math"] >= 5) & (students["Physics"] >= 5), "Đạt", "Thi lại")
print(students)

# Assignment 3:
# Hãy viết tiếp đoạn chương trình của bài 2 để thêm 1 cột dữ liệu tính tổng điểm của 2 môn học
students["Tổng Điểm"] = students["Math"] + students["Physics"]
print(students)

# Assignment 4
total = students.iloc[:, 4].values
total = total.reshape(1, -1)
print(total)
# For total, Let threshold be 10
result = Binarizer(10)
# transformed feature
print(result.fit_transform(total))

# Assignment 5
features = np.array([[50, 50],
                     [49, 50],
                     [48, 49],
                     [-1.83, 3.52],
                     [-2.76, 5.55],
                     [-7.57, 4.90],
                     [-1.85, 3.51],
                     [-7.587, 3.72],
                     [-17, -15],
                     [-1.78, 3.47],
                     [-1.98, 4.022],
                     [-1.97, 2.34],
                     [-5.25, 3.30],
                     [-2.35, 4.0],
                     [2.42, 5.14],
                     [-1.61, 4.989],
                     [-2.18, 3.33],
                     [-20, -18],
                     [-20, -20],
                     [-21, -19]])
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])
clusterer = KMeans(3, random_state=0)
# fit clusterer
clusterer.fit(features)
# Predict values
dataframe["group"] = clusterer.predict(features)
# View first few observation
result = dataframe.head(20)
print(result)

# plot
a = []
b = []
for i in range(0, 20):
    a.append(features[i, 0])
    b.append(features[i, 1])

plt.plot(a, b, "ro")
plt.show()

# Assignment 6
# Hãy gõ đoạn chương trình sau đây và điền câu lệnh vào chỗ trống để thu được kết quả xóa dữ liệu lỗi (missing values).
features = np.array([[12, 12.1],
                     [22.2, 222.2],
                     [44.4, 444.4],
                     [np.nan, 555]])

print(features[~np.isnan(features).any(axis=1)])
# Assignment 7
# Create feature matrix with categorical feature
X = np.array([[1, 4.10, 4.45],
              [2, -3.18, -3.33],
              [2, -3.22, -3.27],
              [1, 4.21, 4.19]])
# Create feature matrix with missing value in the categorical feature
X_with_nan = np.array([[np.nan, 1.87, 1.31],
                       [np.nan, -2.67, -2.22]])
# train KNN learner
clf = KNeighborsClassifier(3, weights="distance")
trained_model = clf.fit(X[:, 1:], X[:, 0])
# Predict missing value'class
imputed_values = trained_model.predict(X_with_nan[:, 1:])
# Join column of predicted class with their other features
X_with_imputed = np.hstack((imputed_values.reshape(-1, 1), X_with_nan[:, 1:]))

# Join two feature matrices
results = np.vstack((X_with_imputed, X))
print(results)
