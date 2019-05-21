import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd

raw_data = loadmat('bruteforce_data.mat')
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']

class_pos = data[data['y'].isin(['1'])]
class_neg = data[data['y'].isin(['0'])]

# plt.scatter(class_pos['X1'], class_pos['X2'], marker='x', label='Positive')
# plt.scatter(class_neg['X1'], class_neg['X2'], marker='o', c='r', s=20, label='Negative')
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

vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

fig, ax = plt.subplots(figsize=(12, 8))

best_score = 0
worst_score = 1

for y in vals:
    for c in vals:
        svc = svm.SVC(kernel='rbf', gamma=y, C=c, probability=True)
        svc.fit(data[['X1', 'X2']], data['y'])

        score = svc.score(data[['X1', 'X2']], data['y'])
        # y_pred = svc.predict(data[['X1', 'X2']])
        # score = accuracy_score(data['y'], y_pred)

        best_score = max(score, best_score)
        worst_score = min(score, worst_score)

        if best_score == score:
            best_gamma = y
            best_C = c

        if worst_score == score:
            worst_gamma = y
            worst_C = c

svc = svm.SVC(kernel='rbf', gamma=best_gamma, C=best_C, probability=True)
svc.fit(data[['X1', 'X2']], data['y'])

Z = svc.decision_function(xy).reshape(XX.shape)
data['Probability'] = svc.predict_proba(data[['X1', 'X2']])[:, 0]
ax.scatter(data['X1'], data['X2'], s=50, c=data['Probability'], cmap='Reds')
ax.contour(XX, YY, Z, colors='b', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

plt.show()

svc = svm.SVC(kernel='rbf', gamma=worst_gamma, C=worst_C, probability=True)
svc.fit(data[['X1', 'X2']], data['y'])

Z = svc.decision_function(xy).reshape(XX.shape)
data['Probability'] = svc.predict_proba(data[['X1', 'X2']])[:, 0]
ax.scatter(data['X1'], data['X2'], s=50, c=data['Probability'], cmap='Reds')
ax.contour(XX, YY, Z, colors='m', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

plt.show()