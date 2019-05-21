import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_predict as cvp
import pandas
import matplotlib.pyplot as plt

data = pandas.read_csv('transit_demands.csv')

data_X = data.values[:, 0:4]
data_y = data.values[:, 4]

# ready data for normalization
data_X[:, 0] = [int(element[1:].strip()) for element in data_X[:, 0]]
data_X[:, 1] = [int(element.replace(',', '')) for element in data_X[:, 1]]
data_X[:, 2] = [int(element[1:].replace(',', '')) for element in data_X[:, 2]]
data_X[:, 3] = [int(element[1:].strip()) for element in data_X[:, 3]]
data_y = [int(element.replace(',', '')) for element in data_y]

# normalizing data
data_X[:, 0] = [element/sum(data_X[:, 0]) for element in data_X[:, 0]]
data_X[:, 1] = [element/sum(data_X[:, 1]) for element in data_X[:, 1]]
data_X[:, 2] = [element/sum(data_X[:, 2]) for element in data_X[:, 2]]
data_X[:, 3] = [element/sum(data_X[:, 3]) for element in data_X[:, 3]]

# adding a column of ones for x0
ones = (np.ones((len(data_X), 1), dtype=float))
data_X = np.concatenate((ones, data_X), axis=1)

data_X_train = data_X[:-3]
data_X_test = data_X[-3:]
data_y_train = data_y[:-3]
data_y_test = data_y[-3:]

model_1 = lm.LinearRegression()

model_1.fit(data_X_train, data_y_train)

mean_square_error = np.mean((model_1.predict(data_X_test) - data_y_test)**2)
variance_coefficient = model_1.score(data_X_test, data_y_test)

print(mean_square_error, variance_coefficient)

X_train, X_test, y_train, y_test = tts(data_X, data_y, test_size=0.1, random_state=0)

model_2 = lm.LinearRegression()
model_2.fit(X_train, y_train)

mean_square_error = np.mean((model_2.predict(X_test) - y_test)**2)
variance_coefficient = model_2.score(X_test, y_test)

print(mean_square_error, variance_coefficient)

y_pred = cvp(model_1, data_X, data_y, cv=10)

fig, ax = plt.subplots()
ax.scatter(data_y, y_pred, edgecolors=(0, 0, 0))
ax.plot([np.min(data_y), np.max(data_y)], [np.min(data_y), np.max(data_y)], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()