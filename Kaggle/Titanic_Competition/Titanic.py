import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import logistic as lr
from sklearn import svm
from sklearn.metrics import accuracy_score as acs
import sklearn.model_selection as ms

raw_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

raw_out = raw_data['Survived']
raw_data = raw_data.drop(['Survived'], 1)

raw_data = pd.concat([raw_data, test_data])

# drop useless columns
raw_data = raw_data.drop(['Name', 'Ticket', 'Cabin'], 1)

# fill nan values in column 'Age' and normalize
rand_min = raw_data['Age'].mean() - raw_data['Age'].std()
rand_max = raw_data['Age'].mean() + raw_data['Age'].std()

np.random.seed(157)
rand_list = np.random.randint(rand_min, rand_max, size=raw_data['Age'].isnull().sum())
raw_data['Age'][np.isnan(raw_data['Age'])] = rand_list

age_sum = raw_data['Age'].sum()
raw_data['Age'] = [element/age_sum for element in raw_data['Age']]

# fill nan values in column 'Fare'
raw_data['Fare'] = raw_data['Fare'].fillna(raw_data['Fare'].mean())

# fill nan values in column 'Embarked', discretize and binarize
raw_data['Embarked'] = raw_data['Embarked'].fillna(method='ffill')
raw_data['Embarked'] = raw_data['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])

Embarked_indices = {}
Embarked = {}
for X in range(0, 3):
    Embarked_indices[X] = raw_data.loc[raw_data['Embarked'] == X+1].values[:, 0] - 1
    Embarked_indices[X] = [int(element) for element in Embarked_indices[X]]
    Embarked[X] = np.zeros([len(raw_data['Embarked'].values), 1], dtype=int)
    Embarked[X][Embarked_indices[X]] = 1

raw_data = raw_data.drop(['Embarked'], 1)
raw_data['Embarked_1'] = Embarked[0]
raw_data['Embarked_2'] = Embarked[1]
raw_data['Embarked_3'] = Embarked[2]

# discretize column 'Sex'
raw_data['Sex'] = raw_data['Sex'].replace(['male', 'female'], [0, 1])

# normalize column 'Fare'
fare_sum = raw_data['Fare'].sum()
raw_data['Fare'] = [element/fare_sum for element in raw_data['Fare']]

# binarize column 'Pclass'
Pclass_indices = {}
Pclass = {}
for X in range(0, 3):
    Pclass_indices[X] = raw_data.loc[raw_data['Pclass'] == X+1].values[:, 0] - 1
    Pclass_indices[X] = [int(element) for element in Pclass_indices[X]]
    Pclass[X] = np.zeros([len(raw_data['Pclass'].values), 1], dtype=int)
    Pclass[X][Pclass_indices[X]] = 1

raw_data = raw_data.drop(['Pclass'], 1)
raw_data['Pclass_1'] = Pclass[0]
raw_data['Pclass_2'] = Pclass[1]
raw_data['Pclass_3'] = Pclass[2]

# binarize column 'SibSp'
SibSp_indices = {}
SibSp = {}
for X in range(0, 6):
    SibSp_indices[X] = raw_data.loc[raw_data['SibSp'] == X].values[:, 0] - 1
    SibSp_indices[X] = [int(element) for element in SibSp_indices[X]]
    SibSp[X] = np.zeros([len(raw_data['SibSp'].values), 1], dtype=int)
    SibSp[X][SibSp_indices[X]] = 1

raw_data = raw_data.drop(['SibSp'], 1)
raw_data['SibSp_0'] = SibSp[0]
raw_data['SibSp_1'] = SibSp[1]
raw_data['SibSp_2'] = SibSp[2]
raw_data['SibSp_3'] = SibSp[3]
raw_data['SibSp_4'] = SibSp[4]
raw_data['SibSp_5'] = SibSp[5]

# binarize column 'Parch'
Parch_indices = {}
Parch = {}
for X in range(0, 6):
    Parch_indices[X] = raw_data.loc[raw_data['Parch'] == X].values[:, 0] - 1
    Parch_indices[X] = [int(element) for element in Parch_indices[X]]
    Parch[X] = np.zeros([len(raw_data['Parch'].values), 1], dtype=int)
    Parch[X][Parch_indices[X]] = 1

raw_data = raw_data.drop(['Parch'], 1)
raw_data['Parch_0'] = Parch[0]
raw_data['Parch_1'] = Parch[1]
raw_data['Parch_2'] = Parch[2]
raw_data['Parch_3'] = Parch[3]
raw_data['Parch_4'] = Parch[4]
raw_data['Parch_5'] = Parch[5]


data_X_train = raw_data.values[:891, 2:]
data_y_train = raw_out.values

data_X_test = raw_data.values[891:, 2:]


# logistic regression with 10-fold cross-validation gives 70.18% accuracy
clf = lr.LogisticRegression(solver='lbfgs')
scores = ms.cross_val_score(clf, data_X_train, data_y_train, cv=10)

print('Logistic regression:', round(scores.mean() * 100, 2), '%')

# grid search for gamma and C
#
# vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
# best_score = 0
#
# for y in vals:
#     for c in vals:
#         print(y, c)
#         clf = svm.SVC(kernel='rbf', gamma=y, C=c)
#         scores = ms.cross_val_score(clf, data_X_train, data_y_train, cv=10)
#         best_score = max(best_score, scores.mean())
#         if best_score == scores.mean():
#             best_gamma = y
#             best_C = c

# SVM with 10-fold cross-validation gives 71.29% accuracy with gaussian kernel
clf = svm.SVC(kernel='rbf', gamma=3000, C=7000)
scores = ms.cross_val_score(clf, data_X_train, data_y_train, cv=10)

print('SVM:', round(scores.mean() * 100, 2), '%')
