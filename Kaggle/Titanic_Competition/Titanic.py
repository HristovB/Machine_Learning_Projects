import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

matplotlib.use('TkAgg')
np.random.seed(157)
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def set_seed(seed_value):

    import os
    import random
    import tensorflow as tf

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from tensorflow.python.keras import backend as k

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                            inter_op_parallelism_threads=1)

    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    k.set_session(sess)


def correlation_plot(X, y, method='feature'):

    # ----- Correlation plot ----- #

    if method == 'feature':

        # Plot feature correlation
        corr = X.corr()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)

        ticks = np.arange(0, len(X.columns), 1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)

        ax.set_yticks(ticks)
        ax.set_xticklabels(X.columns)
        ax.set_yticklabels(X.columns)

        plt.savefig('feature_correlation.png')
        plt.show()

    elif method == 'label':

        # Plot label correlation
        from sklearn.feature_selection import mutual_info_classif

        data_mutual_info = mutual_info_classif(X=X, y=y, random_state=157)

        plt.subplots(1, figsize=(12, 8))
        sns.heatmap(data_mutual_info[:, np.newaxis].T, cmap='Blues', cbar=False, linewidths=1, annot=True)
        plt.yticks([], [])
        plt.gca().set_xticklabels(X.columns[0:], rotation=45, ha='right', fontsize=12)
        plt.suptitle("mutual_info_classif)", fontsize=18, y=1.2)
        plt.gcf().subplots_adjust(wspace=0.2)
        plt.savefig('output_correlation.png')
        plt.show()

    elif method != 'feature' and method != 'label':
        raise ValueError('Wrong input for method! Acceptable inputs: \'feature\', \'label\'')


def encode_values(values):
    lb = LabelEncoder()
    return lb.fit_transform(values)


def preprocessing(data):

    # Drop columns that do not carry valuable information

    data = data.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1)

    # Drop the passengers that do not have values for 'Embarked' column (the rows are dropped and not filled because there are very few of them so they won't affect the classification)
    data = data.drop(data.loc[data['Embarked'].isnull()].index)

    # Encode values of 'Sex' and 'Embarked' columns
    data['Sex'] = encode_values(data['Sex'])
    data['Embarked'] = encode_values(data['Embarked'])

    # Extract 'Title' feature containing only the title of the passenger from the existing 'Name' column
    data['Title'] = [value.split(', ')[1].split('.')[0] for value in data['Name'].values]

    # Extract 'Alone' feature from the existing 'SibSp' and 'Parch' columns
    data['Alone'] = np.zeros(data.shape[0], dtype=np.int)
    data['Alone'].loc[(data['SibSp'] == 0) & (data['Parch'] == 0)] = 1

    # Find mean values of 'Age' for each 'Title' variable
    mean_ages_title = data[['Title', 'Age']].loc[data['Age'].notna()].groupby('Title').mean()
    age_nan_titles = data['Title'].loc[data['Age'].isnull()].unique()

    # Fill 'Age' values that are missing with mean value for each 'Title' class
    for i in range(len(age_nan_titles)):
        if age_nan_titles[i] == 'Mr' or age_nan_titles[i] == 'Mrs' or age_nan_titles[i] == 'Miss':
            data['Age'].loc[(data['Age'].isnull()) & (data['Title'] == age_nan_titles[i])] = [np.ceil(mean_ages_title.loc[age_nan_titles[i]].values[0]) + np.random.randint(-6, 6)
                                                                                              for _ in data['Age'].loc[(data['Age'].isnull()) & (data['Title'] == age_nan_titles[i])]]
        else:
            data['Age'].loc[(data['Age'].isnull()) & (data['Title'] == age_nan_titles[i])] = [np.ceil(mean_ages_title.loc[age_nan_titles[i]].values[0])
                                                                                              for _ in data['Age'].loc[(data['Age'].isnull()) & (data['Title'] == age_nan_titles[i])]]

    # Extract 'AgeRange' feature to transform the numerical 'Age' column to categorical
    _, age_categories = pd.cut(data['Age'], 5, retbins=True)
    age_categories = [np.floor(value) for value in age_categories]

    data['AgeRange'] = np.zeros(len(data), dtype=np.int)

    data['AgeRange'].loc[data['Age'] <= age_categories[1]] = 0
    data['AgeRange'].loc[(data['Age'] > age_categories[1]) & (data['Age'] <= age_categories[2])] = 1
    data['AgeRange'].loc[(data['Age'] > age_categories[2]) & (data['Age'] <= age_categories[3])] = 2
    data['AgeRange'].loc[(data['Age'] > age_categories[3]) & (data['Age'] <= age_categories[4])] = 3
    data['AgeRange'].loc[data['Age'] > age_categories[4]] = 4

    # Extract 'FareRange' feature to transform the numerical 'Fare' column to categorical
    _, fare_categories = pd.qcut(data['Fare'], 4, retbins=True)

    data['FareRange'] = np.zeros(len(data), dtype=np.int)
    data['FareRange'].loc[data['Fare'] <= fare_categories[1]] = 0
    data['FareRange'].loc[(data['Fare'] > fare_categories[1]) & (data['Fare'] <= fare_categories[2])] = 1
    data['FareRange'].loc[(data['Fare'] > fare_categories[2]) & (data['Fare'] <= fare_categories[3])] = 2
    data['FareRange'].loc[data['Fare'] > fare_categories[3]] = 3

    # Sum rare titles into one title called 'Other'
    data['Title'].loc[(data['Title'] != 'Mr') & (data['Title'] != 'Mrs') & (data['Title'] != 'Miss') & (data['Title'] != 'Master')] = 'Other'
    data['Title'] = encode_values(data['Title'])

    # Extract 'FamilyCount' feature
    data['FamilyCount'] = data['SibSp'] + data['Parch']

    # Drop the 'Name' column as it is no longer useful
    data = data.drop('Name', axis=1)

    # Separate features and classes
    y_data = data.get('Survived')
    x_data = data.drop(['Age', 'Fare', 'SibSp', 'Parch'], axis=1)

    scaler = StandardScaler()
    column_names = x_data.columns
    x_data = pd.DataFrame(scaler.fit_transform(x_data), columns=column_names)

    return x_data, y_data


from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from keras.models import load_model


# noinspection DuplicatedCode
def neural_network():
    from keras.models import Sequential
    from keras.layers import BatchNormalization, Activation, Dropout, Dense
    from keras.initializers import he_uniform, glorot_uniform
    from keras.activations import relu, tanh
    from keras.constraints import maxnorm
    from keras.regularizers import l2

    init_relu = he_uniform(seed=157)        # used for ReLU
    init_tanh = glorot_uniform(seed=157)    # used for tanh and softmax

    model = Sequential()

    # input and first Convolutional layer

    model.add(Dense(units=8, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))
    # model.add(Dropout(0.10, seed=157))

    model.add(Dense(units=128, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))
    # model.add(Dropout(0.15, seed=157))

    # second Convolutional layer

    model.add(Dense(units=256, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))
    # model.add(Dropout(0.1, seed=157))

    # second Convolutional layer

    model.add(Dense(units=512, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))
    # model.add(Dropout(0.10, seed=157))

    # second Convolutional layer

    model.add(Dense(units=1024, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))
    # model.add(Dropout(0.15, seed=157))

    # second Convolutional layer

    model.add(Dense(units=512, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))
    # model.add(Dropout(0.10, seed=157))

    # second Convolutional layer

    model.add(Dense(units=256, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))
    # model.add(Dropout(0.1, seed=157))

    # second Convolutional layer

    model.add(Dense(units=128, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))
    # model.add(Dropout(0.15, seed=157))

    model.add(Dense(units=64, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))
    # model.add(Dropout(0.10, seed=157))

    # softmax output Dense layer

    model.add(Dense(units=1, activation='sigmoid', kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))

    return model


def train_network(train_data, train_classes, val_data, val_classes, seed, learning_rate, early_stopping=False):
    from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
    from keras.optimizers import Adam
    from keras.models import save_model
    from keras.metrics import BinaryCrossentropy

    set_seed(seed_value=seed)

    opt = Adam(learning_rate=learning_rate)

    clf = neural_network()

    clf.compile(optimizer=opt,
                loss='binary_crossentropy',
                metrics=['accuracy'])

    checkpoint = ModelCheckpoint('neural_network_checkpoint_training.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    if early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    else:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=1, mode='auto')

    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    clf.fit(x=train_data, y=train_classes,
            epochs=1000,
            validation_data=(val_data, val_classes),
            batch_size=200,
            shuffle=True,
            callbacks=[tensorboard, checkpoint, early_stopping],
            verbose=1)

    save_model(checkpoint.model, 'neural_network_latest_saved.h5')


if __name__ == '__main__':

    # Read dataset into variable
    training_data = pd.read_csv('train.csv')
    testing_data = pd.read_csv('test.csv')

    X_train, y_train = preprocessing(training_data)
    X_train = X_train.drop('Survived', axis=1)

    X_test, y_test = preprocessing(testing_data)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=157, stratify=y_train)

    train_network(train_data=X_train, train_classes=y_train, val_data=X_val, val_classes=y_val, seed=157, learning_rate=0.0001, early_stopping=False)

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
    #     # clf = ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=157, max_features=X_train.shape[1], max_depth=1, bootstrap=True)
    #
    #     clf = XGBClassifier(n_estimators=150, n_jobs=-1, random_state=157, max_depth=1, learning_rate=0.1)
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
    # #
    #

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
