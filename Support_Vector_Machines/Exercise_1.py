import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy.io import loadmat
import pandas as pd

# read data
raw_data = loadmat('nonlin_data.mat')

data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']

class_positive = data[data['y'].isin([1])]
class_negative = data[data['y'].isin([0])]

# plot data points
# plt.scatter(class_positive['X1'], class_positive['X2'], marker='x', label='Positive [1]')
# plt.scatter(class_negative['X1'], class_negative['X2'], marker='o', s=20, c='r', label='Negative [0]')
# plt.legend()
# plt.show()

x1_min = min(data['X1'])
x1_max = max(data['X1'])

x2_min = min(data['X2'])
x2_max = max(data['X2'])

xx = np.linspace(x1_min, x1_max, 30)
yy = np.linspace(x2_min, x2_max, 30)

YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

svc = svm.SVC(kernel='rbf', C=10000, gamma=0.1, probability=True)
svc.fit(data[['X1', 'X2']], data['y'])

Z = svc.decision_function(xy).reshape(XX.shape)

data['Probability'] = svc.predict_proba(data[['X1', 'X2']])[:, 0]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data['X1'], data['X2'], s=30, c=data['Probability'], cmap='Reds')
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.show()