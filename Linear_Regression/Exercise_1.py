import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()

fig, splt = plt.subplots(2, 1)

for dsize in range(-441, -41, 1):
    diabetes_X_train = diabetes.data[:dsize]
    diabetes_X_test = diabetes.data[dsize:]
    diabetes_y_train = diabetes.target[:dsize]
    diabetes_y_test = diabetes.target[dsize:]

    rmodel = linear_model.LinearRegression()
    rmodel.fit(diabetes_X_train, diabetes_y_train)

    mean_square_error = np.mean((rmodel.predict(diabetes_X_test) - diabetes_y_test)**2)
    variance_coefficient = rmodel.score(diabetes_X_test, diabetes_y_test)

    print(dsize + 442)
    splt[0].plot((dsize + 442), mean_square_error, '.r-')
    splt[1].plot((dsize + 442), variance_coefficient, '.b-')

# print(rmodel.coef_)
# print(np.mean((rmodel.predict(diabetes_X_test) - diabetes_y_test)**2))
# print(rmodel.score(diabetes_X_test, diabetes_y_test))

plt.show()