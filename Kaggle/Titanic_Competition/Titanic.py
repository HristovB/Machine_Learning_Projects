import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from keras.models import load_model
from Kaggle.Common_Files.utils import set_seed, train_network

matplotlib.use('TkAgg')
np.random.seed(157)
# pd.set_option('mode.chained_assignment', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def encode_values(dataset):
    lb = LabelEncoder()
    return lb.fit_transform(dataset)


def preprocessing(data):

    # Drop columns that do not carry valuable information

    data = data.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1)

    # Drop the passengers that do not have values for 'Embarked' column (the rows are dropped and not filled because there are very few of them so they won't affect the classification)
    data = data.drop(data.loc[data['Embarked'].isnull()].index)

    # Extract 'Title' feature containing only the title of the passenger from the existing 'Name' column
    data['Title'] = [value.split(', ')[1].split('.')[0] for value in data['Name'].values]

    # Extract 'Surname' feature
    data['Surname'] = encode_values([value.split(', ')[0] for value in data['Name'].values])

    # Extract 'Alone' feature from the existing 'SibSp' and 'Parch' columns
    data['Alone'] = np.zeros(data.shape[0], dtype=np.int)
    data.loc[(data['SibSp'] == 0) & (data['Parch'] == 0), 'Alone'] = 1

    data['FamilyCount'] = data['SibSp'] + data['Parch']

    # Find mean values of 'Age' for each 'Title' variable
    mean_ages_title = data[['Title', 'Age']].loc[data['Age'].notna()].groupby('Title').mean()
    age_nan_titles = data['Title'].loc[data['Age'].isnull()].unique()

    # Fill 'Age' values that are missing with mean value for each 'Title' class
    for i in range(len(age_nan_titles)):
        if age_nan_titles[i] == 'Mr' or age_nan_titles[i] == 'Mrs' or age_nan_titles[i] == 'Miss':
            data.loc[(data['Age'].isnull()) & (data['Title'] == age_nan_titles[i]), 'Age'] = [np.ceil(mean_ages_title.loc[age_nan_titles[i]].values[0]) + np.random.randint(-6, 6)
                                                                                              for _ in data['Age'].loc[(data['Age'].isnull()) & (data['Title'] == age_nan_titles[i])]]
        else:
            data.loc[(data['Age'].isnull()) & (data['Title'] == age_nan_titles[i]), 'Age'] = [np.ceil(mean_ages_title.loc[age_nan_titles[i]].values[0])
                                                                                              for _ in data['Age'].loc[(data['Age'].isnull()) & (data['Title'] == age_nan_titles[i])]]

    # Extract 'AgeRange' feature to transform the numerical 'Age' column to categorical
    _, age_categories = pd.cut(data['Age'], 5, retbins=True)
    age_categories = [np.floor(value) for value in age_categories]

    data['AgeRange'] = np.zeros(len(data), dtype=np.int)

    data.loc[data['Age'] <= age_categories[1], 'AgeRange'] = 0
    data.loc[(data['Age'] > age_categories[1]) & (data['Age'] <= age_categories[2]), 'AgeRange'] = 1
    data.loc[(data['Age'] > age_categories[2]) & (data['Age'] <= age_categories[3]), 'AgeRange'] = 2
    data.loc[(data['Age'] > age_categories[3]) & (data['Age'] <= age_categories[4]), 'AgeRange'] = 3
    data.loc[data['Age'] > age_categories[4], 'AgeRange'] = 4

    # Extract 'FareRange' feature to transform the numerical 'Fare' column to categorical
    _, fare_categories = pd.qcut(data['Fare'], 4, retbins=True)

    data['FareRange'] = np.zeros(len(data), dtype=np.int)
    data.loc[data['Fare'] <= fare_categories[1], 'FareRange'] = 0
    data.loc[(data['Fare'] > fare_categories[1]) & (data['Fare'] <= fare_categories[2]), 'FareRange'] = 1
    data.loc[(data['Fare'] > fare_categories[2]) & (data['Fare'] <= fare_categories[3]), 'FareRange'] = 2
    data.loc[data['Fare'] > fare_categories[3], 'FareRange'] = 3

    # Sum rare titles into one title called 'Other'
    data.loc[(data['Title'] == 'Lady') | (data['Title'] == 'Mme') | (data['Title'] == 'Ms') | (data['Title'] == 'the Countess') | (data['Title'] == 'Mlle'), 'Title'] = 'Miss'
    data.loc[(data['Title'] != 'Mr') & (data['Title'] != 'Mrs') & (data['Title'] != 'Miss') & (data['Title'] != 'Master'), 'Title'] = 'Other'

    # One-Hot encode values of certain columns
    data = pd.get_dummies(data=data, columns=['Sex', 'Embarked', 'Pclass', 'AgeRange', 'FareRange', 'Title'], dtype=np.int)

    # Separate features and classes
    y_data = data.get('Survived')
    x_data = data.drop(['Name', 'Age', 'Fare', 'SibSp', 'Parch'], axis=1)

    scaler = StandardScaler()
    column_names = x_data.columns
    x_data = pd.DataFrame(scaler.fit_transform(x_data), columns=column_names)

    return x_data, y_data


# noinspection DuplicatedCode
def neural_network():
    from keras.models import Sequential
    from keras.layers import BatchNormalization, Activation, Dense, Dropout
    from keras.initializers import glorot_uniform, he_uniform
    from keras.activations import tanh
    from keras.constraints import maxnorm

    init_tanh = glorot_uniform(seed=157)    # used for tanh and softmax

    model = Sequential()

    # input and first Dense layer
    model.add(Dense(units=25, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))

    # third Dense layer
    model.add(Dense(units=50, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))

    # third Dense layer
    model.add(Dense(units=100, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))

    # third Dense layer
    model.add(Dense(units=25, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))

    # third Dense layer
    model.add(Dense(units=100, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))

    # softmax output Dense layer
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))

    return model


if __name__ == '__main__':

    # Read dataset into variable
    training_data = pd.read_csv('train.csv')
    testing_data = pd.read_csv('test.csv')

    X_train, y_train = preprocessing(training_data)
    X_train = X_train.drop('Survived', axis=1)

    X_test, y_test = preprocessing(testing_data)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=157, stratify=y_train)

    train_network(train_data=X_train, train_classes=y_train, val_data=X_val, val_classes=y_val, clf=neural_network(), seed=157, learning_rate=0.0009, epochs=5000, batch_size=50,
                  loss_function='binary_crossentropy', metric='accuracy', early_stopping=False)

    # cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=157)
    #
    # score = []
    # best_clf = None
    # for train_index, test_index in cv.split(X_train, y_train):
    #     # clf = KNeighborsClassifier(n_neighbors=2, n_jobs=-1, weights='distance', metric='manhattan')
    #
    #     # clf = SVC(kernel='rbf', gamma='auto', C=5, random_state=157)
    #
    #     # clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=150, random_state=157, verbose=True)
    #
    #     # clf_base = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=157, max_depth=1)
    #     # clf = AdaBoostClassifier(base_estimator=clf_base, n_estimators=10, random_state=157)
    #
    #     clf = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=157, max_depth=10, bootstrap=True)
    #
    #     # clf = XGBClassifier(n_estimators=1500, tree_method='exact', learning_rate=0.01, colsample_bynode=0.5, colsample_bytree=0.5, colsample_bylevel=0.5,
    #     #                     booster='gbtree', reg_lambda=0.01, verbosity=1, random_state=157, n_jobs=-1)
    #
    #     train_data, val_data = X_train.values[train_index], X_train.values[test_index]
    #     train_output, val_output = y_train.values[train_index], y_train.values[test_index]
    #
    #     clf.fit(train_data, train_output)
    #
    #     val_prediction = clf.predict(val_data)
    #
    #     score.append(f1_score(val_output, val_prediction))
    #
    #     print('F1-score:', score[-1])
    #
    #     if len(score) == 1:
    #         best_clf = clf
    #
    #     elif len(score) > 1 and score[-1] > max(score[:-1]):
    #         best_clf = clf
    #
    # print('Best score:', max(score))
    # print('Average score:', np.mean(score))

    # best_clf = load_model('neural_network_checkpoint_training.h5')
    # threshold = 0.5
    #
    # test_prediction = best_clf.predict(X_test.values).flatten()
    #
    # test_prediction[test_prediction <= threshold] = 0
    # test_prediction[test_prediction > threshold] = 1
    #
    # test_prediction = pd.DataFrame(test_prediction, columns=['Survived'], dtype=np.int)
    #
    # prediction = pd.concat([testing_data['PassengerId'], test_prediction], axis=1)
    # prediction.to_csv('kaggle_submission_nn.csv', index=False)
