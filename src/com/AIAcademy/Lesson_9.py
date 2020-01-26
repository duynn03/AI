import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures

warnings.simplefilter(action='ignore', category=FutureWarning)

matplotlib.style.use('ggplot')

# Assignment 1
df = pd.DataFrame({
    # positive skew
    'x1': np.random.chisquare(8, 1000),
    # negative skew
    'x2': np.random.beta(8, 2, 1000) * 40,
    # no skew
    'x3': np.random.normal(50, 3, 1000)})

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=['x1', 'x2', 'x3'])

_, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'], ax=ax1)
sns.kdeplot(df['x2'], ax=ax1)
sns.kdeplot(df['x3'], ax=ax1)
ax1.set_title('After Min-Max Scaling')
sns.kdeplot(scaled_df['x1'], ax=ax2)
sns.kdeplot(scaled_df['x2'], ax=ax2)
sns.kdeplot(scaled_df['x3'], ax=ax2)
plt.show()

# Assignment 2
df = pd.DataFrame({
    'x1': np.random.normal(0, 2, 1000),
    'x2': np.random.normal(5, 3, 1000),
    'x3': np.random.normal(-5, 5, 1000)})

scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=['x1', 'x2', 'x3'])
print(round(scaled_df.mean()))

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'], ax=ax1)
sns.kdeplot(df['x2'], ax=ax1)
sns.kdeplot(df['x3'], ax=ax1)
ax1.set_title('After Min-Max Scaling')
sns.kdeplot(scaled_df['x1'], ax=ax2)
sns.kdeplot(scaled_df['x2'], ax=ax2)
sns.kdeplot(scaled_df['x3'], ax=ax2)
plt.show()

# Assignment 3


# Assignment 4
x = [[1.0, 2.0],
     [3.0, 4.0],
     [5.0, 6.0]]

polynomialFeatures_interaction = PolynomialFeatures(degree=2, include_bias=True)
result = polynomialFeatures_interaction.fit_transform(x)
print(result)

# Assignment 5
features = np.array([[0.2, 0.3, 0.4],
                     [0.1, 0.3, 0.5],
                     [1, 0.25, 0.9]])


def transformFunction(x):
    return np.round(x * 255)


# create transform
transformer = FunctionTransformer(transformFunction)
# transform feature matrix
result = transformer.transform(features)
print(result)

# Assignment 6
features = np.array([[50, 50],
                     [-2.33, 4.39],
                     [-7.94, 2.10],
                     [-1.83, 3.52],
                     [-2.76, 5.55],
                     [-7.57, 4.90],
                     [-1.85, 3.51],
                     [-7.587, 3.72],
                     [8.52, 3.64],
                     [-1.78, 3.47],
                     [-1.98, 4.022],
                     [-1.97, 2.34],
                     [-5.25, 3.30],
                     [-2.35, 4.0],
                     [2.42, 5.14],
                     [-1.61, 4.989],
                     [-2.18, 3.33],
                     [2, 4],
                     [-20, -20],
                     [-2.77, 4.64]])

# Create detector
outlier_detector = EllipticEnvelope(contamination=.1)

# fit detector
outlier_detector.fit(features)
result = outlier_detector.predict(features)
print(result)
