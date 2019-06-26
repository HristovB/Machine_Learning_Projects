import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import IsolationForest
# from tsfresh.feature_extraction import extract_features
# from tsfresh.utilities.dataframe_functions import impute
# from tsfresh.feature_selection import select_features
# from tsfresh.feature_extraction.settings import EfficientFCParameters, MinimalFCParameters
# from sklearn.decomposition import PCA


def set_seed():
    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value = 157

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random

    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np

    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf

    tf.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as K

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def get_datetime(data):
    # separate datetime column from other data for easier setup
    datetime = data.get(['DATETIME'])

    # setup datetime column
    dt_vals = datetime.values
    dt_vals = dt_vals.flatten()
    date, time = zip(*(element.split(' ') for element in dt_vals))
    day, month, _ = zip(*(element.split('/') for element in date))

    time = np.asarray(time)
    day = np.asarray(day)
    month = np.asarray(month)

    time = np.asarray([element.lstrip('0') for element in time])
    time[time == ''] = '0'

    day = np.asarray([element.lstrip('0') for element in day])
    month = np.asarray([element.lstrip('0') for element in month])

    time = time.astype(np.int)
    day = day.astype(np.int)
    month = month.astype(np.int)

    datetime = pd.DataFrame({'Time_Sine': np.transpose(np.sin(time)), 'Time_Cosine': np.transpose(np.cos(time)), 'Day_Sine': np.transpose(np.sin(day)), 'Day_Cosine': np.transpose(np.cos(day)), 'Month_Sine': np.transpose(np.sin(month)), 'Month_Cosine': np.transpose(np.cos(month))})

    return datetime


def estimate_gaussian(data):  # calculate the mean and covariance of a dataset
    m, n = data.shape

    mean = data.mean(axis=0)  # mean of the data

    cov = np.cov(data, rowvar=False)

    # data = (data.values - mean.values.reshape(1, n))  # calculates (X - u)
    # # cov = np.transpose(data).dot(data)
    # # cov = cov * (1.0/m)  # covariance of the data

    return mean, cov


def multivariate_gaussian_distribution(data, mean, cov):  # create the multivariate gaussian function
    m, n = data.shape
    p = 1.0 / ((2 * np.pi)**(n/2) * np.linalg.det(cov)**0.5) * np.exp(-(0.5 * (data - mean).dot(np.linalg.inv(cov)).dot((data - mean).T)))

    return p


def selectThreshold(y_true, p):
    y_true = y_true.values.flatten()

    best_epsilon = 0.0
    best_f1 = 0.0

    stepsize = (max(p) - min(p)) / 1000
    for epsilon in np.arange(min(p), max(p), stepsize):
        y_pred = (p < epsilon).astype(int)

        tp = np.sum((y_pred == 1).astype(int) & (y_true == 1).astype(int))
        tn = np.sum((y_pred == 0).astype(int) & (y_true == 0).astype(int))
        fp = np.sum((y_pred == 1).astype(int) & (y_true == 0).astype(int))
        fn = np.sum((y_pred == 0).astype(int) & (y_true == 1).astype(int))

        rec = tp / (tp + fn)
        prec = tp / (tp + fp)

        f1 = (2 * prec * rec) / (prec + rec)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1, tn, fp, fn, tp


def calculate_f1(y_true, y_pred=None, p=None, epsilon=None):
    y_true = y_true.values.flatten().astype(int)

    if epsilon != None:
        y_pred = (p < epsilon).astype(int)

    print(y_pred)
    tp = np.sum((y_pred == 1).astype(int) & (y_true == 1).astype(int))
    tn = np.sum((y_pred == 0).astype(int) & (y_true == 0).astype(int))
    fp = np.sum((y_pred == 1).astype(int) & (y_true == 0).astype(int))
    fn = np.sum((y_pred == 0).astype(int) & (y_true == 1).astype(int))

    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)

    f1 = (TPR + TNR) / 2

    return f1, tn, fp, fn, tp


def mgd_anomaly_detection(train_data, validation_data, test_data, validation_att, test_att):
    u, sigma = estimate_gaussian(train_data)

    inx = []
    new_sigma = []
    count = 0
    for i in range(train_data.shape[1]):
        new_sigma.append(sigma[:, i])
        rank = np.linalg.matrix_rank(new_sigma)
        slen = len(new_sigma)

        if (rank + count) < slen:
            inx.append(i)
            count = count + 1

    train_data.drop(train_data.iloc[:, inx], axis=1, inplace=True)
    validation_data.drop(validation_data.iloc[:, inx], axis=1, inplace=True)
    test_data.drop(test_data.iloc[:, inx], axis=1, inplace=True)

    u, sigma = estimate_gaussian(train_data)

    p = multivariate_normal.pdf(validation_data, mean=u, cov=sigma)

    epsilon, f1, tn, fp, fn, tp = selectThreshold(validation_att, p=p)

    p = multivariate_normal.pdf(test_data, mean=u, cov=sigma)

    f1, tn, fp, fn, tp = calculate_f1(test_att, p=p, epsilon=epsilon)

    return f1, tn, fp, fn, tp


if __name__ == '__main__':
    # ------------------------------------------------------- MGD ------------------------------------------------------------- #

    train_data = pd.read_csv('dataset03.csv')
    validation_data = pd.read_csv('dataset04.csv')
    test_data = pd.read_csv('dataset05.csv')

    train_dt = get_datetime(train_data)
    val_dt = get_datetime(validation_data)
    test_dt = get_datetime(test_data)

    train_S = train_data.get(['S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9',  'S_PU10', 'S_PU11', 'S_V2'])
    val_S = validation_data.get(['S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9',  'S_PU10', 'S_PU11', 'S_V2'])
    test_S = test_data.get(['S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9',  'S_PU10', 'S_PU11', 'S_V2'])

    train_att = train_data.get(['ATT_FLAG'])
    train_data = train_data.drop(['ATT_FLAG', 'DATETIME'], axis=1)

    validation_att = validation_data.get(['ATT_FLAG'])
    validation_data = validation_data.drop(['ATT_FLAG', 'DATETIME'], axis=1)

    test_att = test_data.get(['ATT_FLAG'])
    test_data = test_data.drop(['ATT_FLAG', 'DATETIME'], axis=1)

    train_data = train_data.drop(['S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9',  'S_PU10', 'S_PU11', 'S_V2'], axis=1)
    validation_data = validation_data.drop(['S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9',  'S_PU10', 'S_PU11', 'S_V2'], axis=1)
    test_data = test_data.drop(['S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9',  'S_PU10', 'S_PU11', 'S_V2'], axis=1)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(train_data)
    train_data.loc[:, :] = scaled_values

    # scale data in range between 0 and 1
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(validation_data)
    validation_data.loc[:, :] = scaled_values

    # scale data in range between 0 and 1
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(test_data)
    test_data.loc[:, :] = scaled_values

    train_data = pd.concat([train_data, train_S], axis=1)
    validation_data = pd.concat([validation_data, val_S], axis=1)
    test_data = pd.concat([test_data, test_S], axis=1)

    # all_data = pd.concat([train_data, validation_data], axis=0, ignore_index=True)
    # data_y = pd.concat([train_att, validation_att], axis=0, ignore_index=True)
    # data_y.columns = ['ATT_FLAG']
    # all_data = pd.concat([all_data, data_y], axis=1)
    # print(all_data)
    # corr = all_data.corr()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(corr, cmap='coolwarm')
    # fig.colorbar(cax)
    # ticks = np.arange(0, len(all_data.columns), 1)
    # ax.set_xticks(ticks)
    # plt.xticks(rotation=90)
    # ax.set_yticks(ticks)
    # ax.set_xticklabels(all_data.columns)
    # ax.set_yticklabels(all_data.columns)
    # plt.show()

    # best_f1 = 0
    # best_tp = 0
    # best_tn = 0
    # best_fn = 0
    # best_fp = 0
    # best_n_features = 0
    # best_fsum = 1000000
    # best_diff = 0
    # f1_sum = []
    # best_avg_f1 = 0
    # avg_f1 = 0
    #
    # for n_features in range(2, train_data.shape[1]):
    #
    #     pca = PCA(n_components=n_features, whiten=True, random_state=157)
    #     pca.fit(train_data)
    #
    #     new_train = pd.DataFrame(pca.transform(train_data))
    #     new_validate = pd.DataFrame(pca.transform(validation_data))
    #     new_test = pd.DataFrame(pca.transform(test_data))
    #
    #     new_train = new_train.loc[:, (new_train != 0).any(axis=0)]
    #     new_validate = new_validate.loc[:, (new_validate != 0).any(axis=0)]
    #     new_test = new_test.loc[:, (new_train != 0).any(axis=0)]
    #
    #     cv = StratifiedKFold(n_splits=10, random_state=157)
    #
    #     for train_index, test_index in cv.split(new_validate, validation_att):
    #         X_train, X_test = new_validate.iloc[train_index, :], new_validate.iloc[test_index, :]
    #         y_train, y_test = validation_att.iloc[train_index, :], validation_att.iloc[test_index, :]
    #         f1, tn, fp, fn, tp = mgd_anomaly_detection(train_data=new_train, validation_data=X_test, test_data=new_test, validation_att=y_test, test_att=test_att)
    #
    #         f_sum = fp + fn
    #         p_sum = tp + tn
    #         diff = p_sum - f_sum
    #
    #         f1_sum.append(f1)
    #
    #         # if f1 > best_f1 or (f1 == best_f1 and best_diff > diff and fn < fp):  #     best_f1 = f1  #     best_fsum = f_sum  #     best_tp = tp  #     best_tn = tn  #     best_fp = fp  #     best_fn = fn  #     best_n_features = n_features  #     best_n_neighbors = n_neighbors
    #
    #     avg_f1 = sum(f1_sum) / len(f1_sum)
    #     print(sum(f1_sum) / len(f1_sum), n_features)
    #
    #     if avg_f1 > best_avg_f1:
    #         best_avg_f1 = avg_f1
    #         best_n_features = n_features
    #         print(best_f1, best_n_features)
    #         print(tn, fp)
    #         print(fn, tp)
    #     print(best_avg_f1, best_n_features)
    #     f1_sum = []

    # ----------------------------------------------------- AUTOENCODER NN -------------------------------------------------------------- #

    from keras.models import Model, load_model
    from keras.layers import Input, Dense
    from keras.callbacks import ModelCheckpoint, TensorBoard
    from keras import regularizers
    import tensorflow as tf

    new_train_data = pd.read_csv('new_train_data.csv')
    new_validation_data = pd.read_csv('new_validation_data.csv')
    new_test_data = pd.read_csv('new_test_data.csv')

    # scale data in range between 0 and 1
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(new_train_data)
    new_train_data.loc[:, :] = scaled_values

    # scale data in range between 0 and 1
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(new_validation_data)
    new_validation_data.loc[:, :] = scaled_values

    # scale data in range between 0 and 1
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(new_test_data)
    new_test_data.loc[:, :] = scaled_values

    new_train_data = pd.concat([new_train_data, pd.DataFrame(np.zeros((new_train_data.shape[0], 1)), columns=['ATT_FLAG'])], 1)
    new_validation_data = pd.concat([new_validation_data, validation_att], 1)
    new_test_data = pd.concat([new_test_data, test_att], 1)

    set_seed()

    input_dim = new_train_data.shape[1]
    encoding_dim = 179
    input_layer = Input(shape=(input_dim, ))

    encoder = Dense(int(encoding_dim * .8), activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim * .6), activation="relu")(encoder)
    decoder = Dense(int(encoding_dim * .6), activation="relu")(encoder)
    decoder = Dense(int(encoding_dim * .8), activation="tanh")(decoder)
    decoder = Dense(input_dim, activation='tanh')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    nb_epoch = 300
    batch_size = 256
    autoencoder.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath="model.h5",
                                   verbose=0,
                                   save_best_only=True)

    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    history = autoencoder.fit(new_train_data, new_train_data,
                              epochs=nb_epoch,
                              batch_size=batch_size,
                              shuffle=True,
                              verbose=1,
                              callbacks=[checkpointer, tensorboard]).history

    # plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper right')
    # plt.show()

    # autoencoder = load_model('model.h5')
    #
    # predictions = autoencoder.predict(new_test_data)
    # mse = np.mean(np.power(new_test_data - predictions, 2), axis=1)
    # error_df = pd.concat([mse, test_att], 1)
    #
    # error_df.columns = ['reconstruction_error', 'true_class']
    #
    # threshold = .84
    #
    # y_pred = np.array([1 if e > threshold else 0 for e in error_df.reconstruction_error.values])
    # conf_matrix = confusion_matrix(error_df.true_class, y_pred)
    # plt.figure(figsize=(12, 12))
    # sns.heatmap(conf_matrix, annot=True, fmt="d")
    # plt.title("Confusion matrix")
    # plt.ylabel('True class')
    # plt.xlabel('Predicted class')
    # plt.show()
    #
    # print(f1_score(y_true=error_df.true_class, y_pred=y_pred))

    # ----------------------------------------------------------- KMeans ------------------------------------------------------------ #

    # new_train_data = pd.read_csv('new_train_data.csv')
    # new_validation_data = pd.read_csv('new_validation_data.csv')
    # new_test_data = pd.read_csv('new_test_data.csv')
    #
    # # scale data in range between 0 and 1
    # scaler = StandardScaler()
    # scaled_values = scaler.fit_transform(new_train_data)
    # new_train_data.loc[:, :] = scaled_values
    #
    # # scale data in range between 0 and 1
    # scaler = StandardScaler()
    # scaled_values = scaler.fit_transform(new_validation_data)
    # new_validation_data.loc[:, :] = scaled_values
    #
    # # scale data in range between 0 and 1
    # scaler = StandardScaler()
    # scaled_values = scaler.fit_transform(new_test_data)
    # new_test_data.loc[:, :] = scaled_values

    # best_fsum = 1000000
    # best_diff = 0
    # best_f1 = 0
    # best_n_features = 0
    # f1_sum = []
    # avg_f1 = 0
    # best_avg_f1 = 0
    # for n_features in range(2, train_data.shape[1]):
    #     pca = PCA(n_components=n_features, whiten=True, random_state=157)
    #     pca.fit(train_data)
    #
    #     new_data_X = pd.concat([pd.DataFrame(pca.transform(train_data)), pd.DataFrame(pca.transform(validation_data))], axis=0, ignore_index=True)
    #     # new_test = pd.DataFrame(pca.transform(new_validation_data))
    #
    #     data_y = pd.concat([train_att, validation_att], axis=0, ignore_index=True)
    #
    #     clf = KMeans(n_clusters=2, random_state=157, max_iter=500, n_init=20)
    #     cv = StratifiedKFold(n_splits=10, random_state=157)
    #
    #     for train_index, test_index in cv.split(new_data_X, data_y):
    #         X_train, X_test = new_data_X.iloc[train_index, :], new_data_X.iloc[test_index, :]
    #         y_train, y_test = data_y.iloc[train_index, :], data_y.iloc[test_index, :]
    #
    #         clf.fit(X_train)
    #         y_pred = clf.predict(X_test)
    #
    #         f1, tn, fp, fn, tp = calculate_f1(y_test, y_pred.flatten())
    #
    #         f_sum = fp + fn
    #         p_sum = tp + tn
    #         diff = p_sum - f_sum
    #
    #         f1_sum.append(f1)
    #
    #     avg_f1 = sum(f1_sum) / len(f1_sum)
    #     print(sum(f1_sum) / len(f1_sum))
    #
    #     if (avg_f1 > best_avg_f1):
    #         best_avg_f1 = avg_f1
    #         best_n_features = n_features
    #         print(best_f1, best_n_features)
    #     print(best_avg_f1, best_n_features)
    #     f1_sum = []

#     pca = PCA(n_components=32, whiten=True, random_state=157)
#     pca.fit(train_data)
#
#     clf = KMeans(n_clusters=2, random_state=157, max_iter=500, n_init=20)
#
#     new_data_X = pd.concat([pd.DataFrame(pca.transform(train_data)), pd.DataFrame(pca.transform(validation_data))], axis=0, ignore_index=True)
#     new_test = pd.DataFrame(pca.transform(test_data))
# 0.
#     clf.fit(X=new_data_X)
#
#     y_pred = clf.predict(new_test)
#
#     y_pred[y_pred == -1] = 0
#
#     f1, tn, fp, fn, tp = calculate_f1(test_att, y_pred.flatten())
#
#     f_sum = fp + fn
#     p_sum = tp + tn
#     diff = p_sum - f_sum
#
#     print(f1)
#     print(tn, fp)
#     print(fn, tp)

    # ----------------------------------------------------- ONE-CLASS SVM --------------------------------------------------------- #

    # best_fsum = 1000000
    # best_diff = 0
    # best_f1 = 0
    # best_n_features = 0
    # f1_sum = []
    # avg_f1 = 0
    # best_avg_f1 = 0
    # for n_features in range(2, 20):
    #     pca = PCA(n_components=n_features, whiten=True, random_state=157)
    #     pca.fit(train_data)
    #
    #     new_data_X = pd.DataFrame(pca.transform(train_data))
    #     new_test = pd.DataFrame(pca.transform(validation_data))
    #
    #     clf = OneClassSVM(gamma='auto', nu=0.2, random_state=157, kernel='sigmoid')
    #
    #     clf.fit(new_data_X)
    #
    #     cv = StratifiedKFold(n_splits=10, random_state=157)
    #
    #     for train_index, test_index in cv.split(new_test, validation_att):
    #         X_train, X_test = new_test.iloc[train_index, :], new_test.iloc[test_index, :]
    #         y_train, y_test = validation_att.iloc[train_index, :], validation_att.iloc[test_index, :]
    #
    #         y_pred = clf.predict(X_test)
    #
    #         f1, tn, fp, fn, tp = calculate_f1(y_test, y_pred.flatten())
    #
    #         f_sum = fp + fn
    #         p_sum = tp + tn
    #         diff = p_sum - f_sum
    #
    #         f1_sum.append(f1)
    #
    #         # if f1 > best_f1 or (f1 == best_f1 and best_diff > diff and fn < fp):  #     best_f1 = f1  #     best_fsum = f_sum  #     best_tp = tp  #     best_tn = tn  #     best_fp = fp  #     best_fn = fn  #     best_n_features = n_features  #     best_n_neighbors = n_neighbors
    #
    #     avg_f1 = sum(f1_sum) / len(f1_sum)
    #     print(sum(f1_sum) / len(f1_sum), n_features)
    #
    #     if (avg_f1 > best_avg_f1):
    #         best_avg_f1 = avg_f1
    #         best_n_features = n_features
    #         print(best_f1, best_n_features)
    #     print(best_avg_f1, best_n_features)
    #     f1_sum = []

    # pca = PCA(n_components=6, whiten=True, random_state=157)
    # pca.fit(train_data)
    #
    # clf = OneClassSVM(gamma='auto', nu=0.2, random_state=157, kernel='sigmoid')
    #
    # new_data_X = pd.DataFrame(pca.transform(train_data))
    # new_test = pd.DataFrame(pca.transform(test_data))
    #
    # clf.fit(X=new_data_X)
    #
    # y_pred = clf.predict(new_test)
    #
    # y_pred[y_pred == -1] = 0
    #
    # f1, tn, fp, fn, tp = calculate_f1(test_att, y_pred.flatten())
    #
    # f_sum = fp + fn
    # p_sum = tp + tn
    # diff = p_sum - f_sum
    #
    # print(f1)
    # print(tn, fp)
    # print(fn, tp)
    #
    # best_fsum = 1000000
    # best_diff = 0
    # best_f1 = 0
    # best_n_features = 0
    # best_gamma = 0
    # best_nu = 0

    # ----------------------------------------------------- ISOLATION FOREST --------------------------------------------------------- #

    # best_fsum = 1000000
    # best_diff = 0
    # best_f1 = 0
    # best_n_features = 0
    # best_n_trees = 0
    # best_nu = 0
    # best_outlier_perc = 0
    # f1_sum = []
    # avg_f1 = 0
    # best_avg_f1 = 0
    #
    # for n_features in range(25, test_data.shape[1]):
    #     for n_trees in range(50, 55):
    #         for outlier_perc in np.arange(start=0.16, stop=0.22, step=0.01):
    #             pca = PCA(n_components=n_features, whiten=True, random_state=157)
    #             pca.fit(test_data)
    #
    #             new_data_X = pd.concat([pd.DataFrame(pca.transform(train_data)), pd.DataFrame(pca.transform(validation_data))], axis=0, ignore_index=True)
    #             data_y = pd.concat([train_att, validation_att], axis=0, ignore_index=True)
    #
    #             # clf = IsolationForest(n_estimators=n_trees, contamination=outlier_perc, bootstrap=True, random_state=157)
    #             cv = StratifiedKFold(n_splits=10, random_state=157)
    #
    #             for train_index, test_index in cv.split(new_data_X, data_y):
    #                 X_train, X_test = new_data_X.iloc[train_index, :], new_data_X.iloc[test_index, :]
    #                 y_train, y_test = data_y.iloc[train_index, :], data_y.iloc[test_index, :]
    #
    #                 clf = IsolationForest(n_estimators=n_trees, contamination=outlier_perc, bootstrap=True, random_state=157, behaviour='new')
    #                 clf.fit(X_train)
    #
    #                 y_pred = clf.predict(X_test)
    #
    #                 f1, tn, fp, fn, tp = calculate_f1(y_test, y_pred.flatten())
    #
    #                 f_sum = fp + fn
    #                 p_sum = tp + tn
    #                 diff = p_sum - f_sum
    #
    #                 f1_sum.append(f1)
    #
    #                 # if f1 > best_f1 or (f1 == best_f1 and best_diff > diff and fn < fp):  #     best_f1 = f1  #     best_fsum = f_sum  #     best_tp = tp  #     best_tn = tn  #     best_fp = fp  #     best_fn = fn  #     best_n_features = n_features  #     best_n_neighbors = n_neighbors
    #
    #             avg_f1 = sum(f1_sum) / len(f1_sum)
    #             print(sum(f1_sum) / len(f1_sum), n_features)
    #
    #             if (avg_f1 > best_avg_f1):
    #                 best_avg_f1 = avg_f1
    #                 best_n_features = n_features
    #                 print(best_f1, best_n_features)
    #             print(best_avg_f1, best_n_features)
    #             f1_sum = []

    # pca = PCA(n_components=6, whiten=True, random_state=157)
    # pca.fit(train_data)
    #
    # clf = IsolationForest(n_estimators=n_trees, contamination=outlier_perc, bootstrap=True, random_state=157)
    #
    # new_data_X = pd.DataFrame(pca.transform(train_data))
    # new_test = pd.DataFrame(pca.transform(test_data))
    #
    # clf.fit(X=new_data_X)
    #
    # y_pred = clf.predict(new_test)
    #
    # y_pred[y_pred == -1] = 0
    #
    # f1, tn, fp, fn, tp = calculate_f1(test_att, y_pred.flatten())
    #
    # f_sum = fp + fn
    # p_sum = tp + tn
    # diff = p_sum - f_sum
    #
    # print(f1)
    # print(tn, fp)
    # print(fn, tp)
    #
    # best_fsum = 1000000
    # best_diff = 0
    # best_f1 = 0
    # best_n_features = 0
    # best_gamma = 0
    # best_nu = 0

    # ---------------------------------------------------------- SVM ---------------------------------------------------------------- #

    # best_fsum = 1000000
    # best_diff = 0
    # best_f1 = 0
    # best_n_features = 0
    # best_gamma = 0
    # best_C = 0
    # best_tn = 0
    # best_fp = 0
    # best_tp = 0
    # best_fn = 0
    #
    # f1_sum = []
    # vals_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
    # for n_features in range(35, train_data.shape[1]):
    #     print(50 * '-', n_features, 50 * '-')
    #     pca = PCA(n_components=n_features, whiten=True, random_state=157)
    #     pca.fit(train_data)
    #
    #     new_data_X = pd.concat([pd.DataFrame(pca.transform(train_data)), pd.DataFrame(pca.transform(validation_data))], axis=0, ignore_index=True)
    #     new_test = pca.transform(test_data)
    #
    #     data_y = pd.concat([train_att, validation_att], axis=0, ignore_index=True)
    #
    #     cv = StratifiedKFold(n_splits=10, random_state=157)
    #
    #     clf = SVC(gamma='auto', random_state=157, kernel='rbf')
    #
    #     for train_index, test_index in cv.split(new_data_X, data_y):
    #         X_train, X_test = new_data_X.iloc[train_index, :], new_data_X.iloc[test_index, :]
    #         y_train, y_test = data_y.iloc[train_index, :], data_y.iloc[test_index, :]
    #
    #         clf.fit(X=X_train, y=y_train)
    #
    #         y_pred = clf.predict(X_test)
    #
    #         y_pred[y_pred == -1] = 0
    #
    #         f1, tn, fp, fn, tp = calculate_f1(y_test, y_pred.flatten())
    #
    #         f_sum = fp + fn
    #         p_sum = tp + tn
    #         diff = p_sum - f_sum
    #
    #         f1_sum.append(f1)
    #         # print(f1, n_features, C)
    #
    #         if f1 > best_f1 or (f1 == best_f1 and best_diff > diff and fn < fp):
    #             best_f1 = f1
    #             best_fsum = f_sum
    #             best_tp = tp
    #             best_tn = tn
    #             best_fp = fp
    #             best_fn = fn
    #             best_n_features = n_features
    #
    #     print(sum(f1_sum)/len(f1_sum))
    #     print(best_f1, best_n_features)
    #     print(best_tn, best_fp)
    #     print(best_fn, best_tp)
    #     f1_sum = []

    # pca = PCA(n_components=31, whiten=True, random_state=157)
    # pca.fit(train_data)
    # clf = SVC(gamma='auto', random_state=157, kernel='rbf')
    #
    # new_data_X = pd.concat([pd.DataFrame(pca.transform(train_data)), pd.DataFrame(pca.transform(validation_data))], axis=0, ignore_index=True)
    # new_test = pca.transform(test_data)
    #
    # data_y = pd.concat([train_att, validation_att], axis=0, ignore_index=True)
    #
    # clf.fit(X=new_data_X, y=data_y)
    #
    # y_pred = clf.predict(new_test)
    #
    # y_pred[y_pred == -1] = 0
    #
    # f1, tn, fp, fn, tp = calculate_f1(test_att, y_pred.flatten())
    #
    # f_sum = fp + fn
    # p_sum = tp + tn
    # diff = p_sum - f_sum
    #
    # print(f1)
    # print(tn, fp)
    # print(fn, tp)

    # ---------------------------------------------------------- KNN ---------------------------------------------------------------- #

    # new_train_data = pd.read_csv('new_train_data.csv')
    # new_validation_data = pd.read_csv('new_validation_data.csv')
    # new_test_data = pd.read_csv('new_test_data.csv')
    #
    # # scale data in range between 0 and 1
    # scaler = StandardScaler()
    # scaled_values = scaler.fit_transform(new_train_data)
    # new_train_data.loc[:, :] = scaled_values
    #
    # # scale data in range between 0 and 1
    # scaler = StandardScaler()
    # scaled_values = scaler.fit_transform(new_validation_data)
    # new_validation_data.loc[:, :] = scaled_values
    #
    # # scale data in range between 0 and 1
    # scaler = StandardScaler()
    # scaled_values = scaler.fit_transform(new_test_data)
    # new_test_data.loc[:, :] = scaled_values
    #
    # best_fsum = 1000000
    # best_diff = 0
    # best_f1 = 0
    # best_n_features = 0
    # best_n_neighbors = 0
    # best_tn = 0
    # best_fp = 0
    # best_tp = 0
    # best_fn = 0
    # f1_sum = []
    # best_avg_f1 = 0
    #
    # for n_features in range(28, train_data.shape[1]):
    #     for n_neighbors in range(2, 10):
    #         pca = PCA(n_components=n_features, whiten=True, random_state=157)
    #         pca.fit(train_data)
    #
    #         new_data_X = pd.concat([pd.DataFrame(pca.transform(train_data)), pd.DataFrame(pca.transform(validation_data))], axis=0, ignore_index=True)
    #         # new_test = pca.transform(test_data)
    #
    #         data_y = pd.concat([train_att, validation_att], axis=0, ignore_index=True)
    #
    #         cv = StratifiedKFold(n_splits=10, random_state=157)
    #         clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    #
    #         for train_index, test_index in cv.split(new_data_X, data_y):
    #             X_train, X_test = new_data_X.iloc[train_index, :], new_data_X.iloc[test_index, :]
    #             y_train, y_test = data_y.iloc[train_index, :], data_y.iloc[test_index, :]
    #
    #             clf.fit(X=X_train, y=y_train.values.flatten())
    #
    #             y_pred = clf.predict(X_test)
    #
    #             y_pred[y_pred == -1] = 0
    #
    #             f1, tn, fp, fn, tp = calculate_f1(y_test, y_pred.flatten())
    #
    #             f_sum = fp + fn
    #             p_sum = tp + tn
    #             diff = p_sum - f_sum
    #
    #             f1_sum.append(f1)
    #
    #             # if f1 > best_f1 or (f1 == best_f1 and best_diff > diff and fn < fp):
    #             #     best_f1 = f1
    #             #     best_fsum = f_sum
    #             #     best_tp = tp
    #             #     best_tn = tn
    #             #     best_fp = fp
    #             #     best_fn = fn
    #             #     best_n_features = n_features
    #             #     best_n_neighbors = n_neighbors
    #
    #         avg_f1 = sum(f1_sum)/len(f1_sum)
    #         print(sum(f1_sum)/len(f1_sum))
    #
    #         if(avg_f1 > best_avg_f1):
    #             best_avg_f1 = avg_f1
    #             best_n_features = n_features
    #             best_n_neighbors = n_neighbors
    #             print(best_f1, best_n_features, best_n_neighbors)
    #             print(best_tn, best_fp)
    #             print(best_fn, best_tp)
    #         print(best_avg_f1, best_n_features, best_n_neighbors)
    #         f1_sum=[]

    # pca = PCA(n_components=28, whiten=True, random_state=157)
    # pca.fit(train_data)
    #
    # clf = KNeighborsClassifier(n_neighbors=3)
    #
    # new_data_X = pd.concat([pd.DataFrame(pca.transform(train_data)), pd.DataFrame(pca.transform(validation_data))], axis=0, ignore_index=True)
    # new_test = pca.transform(test_data)
    #
    # data_y = pd.concat([train_att, validation_att], axis=0, ignore_index=True)
    # clf.fit(X=new_data_X, y=data_y)
    #
    # y_pred = clf.predict(new_test)
    #
    # y_pred[y_pred == -1] = 0
    #
    # f1, tn, fp, fn, tp = calculate_f1(test_att, y_pred.flatten())
    #
    # f_sum = fp + fn
    # p_sum = tp + tn
    # diff = p_sum - f_sum
    #
    # print(f1)
    # print(tn, fp)
    # print(fn, tp)

    # ---------------------------------------------------------- RESULTS ---------------------------------------------------------------- #
    # x = np.array(['kNN', 'SVC', 'K-Means', 'One-Class SVM', 'MGD', 'Isolation Forest', 'Autoencoder'])
    # y = np.array([0.3689, 0.5182, 0.3051, 0.3278, 0.4897, 0.628, 0.8809])
    # sns.set(style="whitegrid", font_scale=2)
    # bar = sns.barplot(x, y)
    # bar.set_ylabel('F1 Score')
    # plt.show()

    # plt.figure(figsize=(12, 12))
    # sns.heatmap(conf_matrix, annot=True, fmt="d")
    # plt.title("Confusion matrix")
    # plt.ylabel('True class')
    # plt.xlabel('Predicted class')
    # plt.show()

# ---------------------------------------------------------------------- TESTING CODE -------------------------------------------------------------- #

# #scale data in range between 0 and 1
# scaler = RobustScaler()
# scaled_values = scaler.fit_transform(train_data)
# train_data.loc[:, :] = scaled_values

# #scale data in range between 0 and 1
# scaler = RobustScaler()
# scaled_values = scaler.fit_transform(validation_data)
# validation_data.loc[:, :] = scaled_values
#
# #scale data in range between 0 and 1
# scaler = RobustScaler()
# scaled_values = scaler.fit_transform(test_data)
# test_data.loc[:, :] = scaled_values

# idx = []
# for i in range(1, 418):
#     for j in range(5):
#         idx.append(i)
#
# # idx = np.ones(train_data.shape[0], dtype=np.int16)
# time = np.arange(test_data.shape[0])
#
# idx = pd.DataFrame({'Id': idx})
# time = pd.DataFrame({'Time': time})
# test_data = pd.concat([idx, time, test_data], 1)
# test_data = test_data[:-4]
#
# # print(test_data)
# settings = EfficientFCParameters()
# f = extract_features(test_data, column_id='Id', column_sort='Time', default_fc_parameters=settings)
# f.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\tsfresh_test_features.csv', index=False)

# print('Reading data...')
# f = pd.read_csv('tsfresh_features_two.csv')
# print('Read data...')
# f = impute(f)
# print('Imputed data...')
# f.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\tsfresh_imputed.csv', index=False)

# print('Reading data...')
# f = pd.read_csv('tsfresh_imputed.csv')
# print('Removing zeroes...')
# f = f.loc[:, (f != 0).any(axis=0)]
# print('Saving data...')
# f.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\tsfresh_clean.csv', index=False)
# print('Done!')

# print('Reading data...')
# f = pd.read_csv('tsfresh_clean.csv')
# num_unique = f.apply(pd.Series.nunique)
# cols_to_drop = num_unique[num_unique < 100].index.tolist()
# print('Removing columns with the same value or mostly zero...')
# f = f.drop(cols_to_drop, axis=1)
# print('Saving data...')
# f.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\tsfresh_clean_two.csv', index=False)
# print('Done!')

# f = pd.read_csv('tsfresh_clean_two.csv')
# df = []
# m, n = f.shape
# for i in range(m):
#     for j in range(5):
#         df.append(f.values[i, :])
#
# df = pd.DataFrame(df)
# df.columns = f.columns
# df.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\tsfresh_non_selected.csv', index=False)

# f = pd.read_csv('tsfresh_non_selected.csv')
#
# pfa = PFA(n_features=150)
# print('Fitting...')
# pfa.fit(f)
# print('Fitting done!')
#
# df = pfa.features_
# column_indices = pfa.indices_
#
# df = pd.DataFrame(df)
# column_indices = pd.DataFrame(column_indices)
#
# print('Saving data...')
# df.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\tsfresh_pfa_features.csv', index=False)
# column_indices.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\tsfresh_pfa_column_indices.csv', index=False)

# f = pd.read_csv('tsfresh_non_selected.csv')
# f_ci = pd.read_csv('tsfresh_pfa_column_indices.csv')
#
# f_new = f.iloc[:, f_ci.values.flatten()]
#
# f_new.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\new_features.csv', index=False)

# f = pd.read_csv('new_features.csv')
#
# train_data = train_data.drop(['Id', 'Time'], axis=1)
# new_train_data = pd.concat([train_data, f], axis=1)
#
# # scale data in range between 0 and 1
# scaler = StandardScaler()
# scaled_values = scaler.fit_transform(new_train_data)
# new_train_data.loc[:, :] = scaled_values
#
# new_train_data.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\new_train_data.csv', index=False)
#
# new_train_data = pd.read_csv('new_train_data.csv')
#
# # .drop(['F_PU1', 'F_PU2', 'F_PU3', 'F_PU4', 'F_PU5', 'F_PU8', 'F_PU9', 'F_PU10', 'F_V2', 'P_J269', 'P_J256', 'P_J289', 'P_J307', 'P_J422'], axis=1)
#
# u, sigma = estimate_gaussian(new_train_data)
#
# print(np.linalg.matrix_rank(sigma))
# print(sigma.shape)
#
# inx = []
# new_sigma = []
# count = 0
# for i in range(167):
#     new_sigma.append(sigma[:, i])
#     rank = np.linalg.matrix_rank(new_sigma)
#     slen = len(new_sigma)
#
#
#     if (rank + count) < slen:
#         inx.append(i)
#         count = count + 1
#
# new_train_data.drop(new_train_data.iloc[:, inx], axis=1, inplace=True)
# u, sigma = estimate_gaussian(new_train_data)
#
# print(np.linalg.matrix_rank(sigma))
# print(sigma.shape)
#
# new_train_data.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\new_train_data.csv', index=False)

# new_train_data = pd.read_csv('new_train_data.csv')
#
# u, sigma = estimate_gaussian(new_train_data)
# p = multivariate_normal.pdf(new_train_data, mean=u, cov=sigma)
# p = pd.DataFrame(p)
# print(p)
# # pd.DataFrame(sigma).to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\sigma.csv', index=False)
#
# pd.DataFrame(p).to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\p.csv', index=False)


# new_train_data = pd.read_csv('new_train_data.csv')
# new_validation_data = pd.read_csv('tsfresh_val_features.csv')
# new_test_data = pd.read_csv('tsfresh_test_features.csv')
#
# df = []
# m, n = new_validation_data.shape
# for i in range(m):
#     for j in range(5):
#         df.append(new_validation_data.values[i, :])
#
# df = pd.DataFrame(df)
# df.columns = new_validation_data.columns
#
# new_validation_data = df
#
# df = []
# m, n = new_test_data.shape
# for i in range(m):
#     for j in range(5):
#         df.append(new_test_data.values[i, :])
#
# df = pd.DataFrame(df)
# df.columns = new_test_data.columns
#
# new_test_data = df
#
# print('Here!')
#
# new_validation_data = pd.concat([validation_data, new_validation_data], axis=1)
# new_test_data = pd.concat([test_data, new_test_data], axis=1)
#
# new_validation_data = new_validation_data[new_train_data.columns.tolist()]
# new_test_data = new_test_data[new_train_data.columns.tolist()]
#
# new_validation_data = impute(new_validation_data)
# new_test_data = impute(new_test_data)
#

#
# new_validation_data.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\new_validation_data.csv', index=False)
# new_test_data.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\new_test_data.csv', index=False)


#
# # scale data in range between 0 and 1
# scaler = StandardScaler()
# scaled_values = scaler.fit_transform(new_validation_data)
# new_validation_data.loc[:, :] = scaled_values
#
# # scale data in range between 0 and 1
# scaler = StandardScaler()
# scaled_values = scaler.fit_transform(new_test_data)
# new_test_data.loc[:, :] = scaled_values

# m, n = new_train_data.shape
# u, sigma = estimate_gaussian(new_train_data)
#
# u = u.values.reshape(1, n)

# print(np.exp(-0.5*((new_train_data.values - u).dot(np.linalg.inv(sigma))).dot((new_train_data.values - u).T)))

# print(-0.5*(new_train_data.values - u).dot(np.linalg.inv(sigma)).dot((new_train_data.values - u).T))

# p = multivariate_normal.pdf(new_validation_data, mean=u, cov=sigma)
#
# pd.DataFrame(p).to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\p.csv', index=False)

# data_X = new_test_data
# data_y = test_att
#
# data_mutual_info = mutual_info_classif(X=data_X, y=data_y, random_state=157)
#
# plt.subplots(1, figsize=(26, 1))
# sns.heatmap(data_mutual_info[:, np.newaxis].T, cmap='Blues', cbar=False, linewidths=1, annot=True)
# plt.yticks([], [])
# plt.gca().set_xticklabels(data_X.columns[0:], rotation=45, ha='right', fontsize=12)
# plt.suptitle("mutual_info_classif)", fontsize=18, y=1.2)
# plt.gcf().subplots_adjust(wspace=0.2)
# plt.show()
# m , n = train_data.shape
#
# for i in range(n):
#     time = np.arange(i*6, (i+1)*6)
#     plt.plot(time, train_data.iloc[i*6:(i+1)*6, 0].values.ravel(), marker='.')
#     plt.xlabel('Time')
#     plt.ylabel(train_data.columns[0])
#     plt.show()
# train_data = data_setup(train_data)
#
# validation_att_flag = validation_data.get(['ATT_FLAG'])
# validation_data = validation_data.drop(['ATT_FLAG'], 1)
# validation_data = data_setup(validation_data)
# validation_data = pd.concat([validation_data, validation_att_flag], 1)
#
# test_att_flag = test_data.get(['ATT_FLAG'])
# test_data = test_data.drop(['ATT_FLAG'], 1)
# test_data = data_setup(test_data)
# test_data = pd.concat([test_data, test_att_flag], 1)

# train_data.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\train_data.csv', index=False)
# validation_data.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\validation_data.csv', index=False)
# test_data.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\test_data.csv', index=False)


# train_data = pd.read_csv('train_data.csv')
# validation_data = pd.read_csv('validation_data.csv')
# test_data = pd.read_csv('test_data.csv')
#
# test_att_data = test_data.get(['ATT_FLAG'])
#
# all_data = pd.concat([train_data])
# # all_data = all_data.drop(['Time_Sine', 'Time_Cosine', 'Day_Sine', 'Day_Cosine', 'Month_Sine', 'Month_Cosine'], 1)
#
# all_data_att = all_data.get(['ATT_FLAG'])
# # all_data = all_data.drop(['ATT_FLAG', 'S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9',  'S_PU10', 'S_PU11', 'S_V2', 'F_PU3', 'F_PU5', 'F_PU9', 'F_PU6', 'F_PU11'], 1)
# all_data = all_data.drop(['ATT_FLAG'], 1)

# 'S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9',  'S_PU10', 'S_PU11', 'S_V2'

# scale data in range between 0 and 1
# scaler = RobustScaler()
# scaled_values = scaler.fit_transform(all_data)
# all_data.loc[:, :] = scaled_values

# all_data.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\all_data.csv', index=False)

# all_data.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\all_data.csv', index=False)

# att_False = all_data.iloc[(all_data_att.values == 0).flatten()]
# att_True = all_data.iloc[(all_data_att.values == 1).flatten()]

# trans = GenericUnivariateSelect(score_func=mutual_info_classif, mode='percentile', param=40)
# trans.fit(data_X, data_y)
# all_data_trans = all_data.loc[:, trans.get_support()]
# all_data_trans = all_data_trans.drop(['S_PU5'], 1)

# data_X = all_data.drop(['ATT_FLAG'], 1)
# data_y = all_data.get(['ATT_FLAG']).values.ravel()

# m, n = data_X.shape
# noise = np.random.normal(0, 0.0001, [m, 1])
# data_X['F_PU3'] = noise
#
# noise = np.random.normal(0, 0.0001, [m, 1])
# data_X['F_PU5'] = noise
#
# noise = np.random.normal(0, 0.0001, [m, 1])
# data_X['F_PU9'] = noise

# scaler = RobustScaler()
# scaled_values = scaler.fit_transform(test_data)
# test_data.loc[:, :] = scaled_values
#
# train_data = all_data.loc[:8760, :]
# validation_data = all_data.loc[8760:, :]
# test_data = test_data.get(all_data.columns.values)

#
# m, n = test_data.shape
#
# noise = np.random.normal(0, 0.0001, [m, 1])
# test_data['F_PU5'] = noise
#
# noise = np.random.normal(0, 0.0001, [m, 1])
# test_data['F_PU9'] = noise
#

# validation_data = pd.concat([validation_data, all_data_att.loc[8760:, :]], 1)

# validation_data_att = pd.DataFrame(all_data_att.loc[8760:, :])

# for i in range(n):
#     for j in range(i+1, n):
#         print(i, j)
#         sns.scatterplot(att_False.iloc[:, i], att_False.iloc[:, j], marker='.')
#         sns.scatterplot(att_True.iloc[:, i], att_True.iloc[:, j], marker='.', color='orange')
#         plt.xlabel(att_True.columns[i])
#         plt.ylabel(att_True.columns[j])
#         # plt.savefig(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\Figures\'' + att_True.columns[i] + 'x' + att_True.columns[j] + '.png', dpi=300)
#         # plt.close()
#         plt.show()

# test_data = pd.concat([test_data, test_att_data], 1)

# input_dim = train_data.shape[1]
# encoding_dim = 40
# input_layer = Input(shape=(input_dim, ))
#
# encoder = Dense(encoding_dim, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
# encoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
# encoder = Dense(int(encoding_dim / 4), activation="tanh")(encoder)
# decoder = Dense(int(encoding_dim / 4), activation="tanh")(encoder)
# decoder = Dense(int(encoding_dim / 2), activation="tanh")(decoder)
# decoder = Dense(input_dim, activation='tanh')(decoder)
# autoencoder = Model(inputs=input_layer, outputs=decoder)
#
# nb_epoch = 300
# batch_size = 30
# autoencoder.compile(optimizer='adam',
#                     loss='mean_squared_error',
#                     metrics=['accuracy'])
#
# checkpointer = ModelCheckpoint(filepath="model.h5",
#                                verbose=0,
#                                save_best_only=True)
#
# tensorboard = TensorBoard(log_dir='./logs',
#                           histogram_freq=0,
#                           write_graph=True,
#                           write_images=True)
#
# history = autoencoder.fit(train_data, train_data,
#                           epochs=nb_epoch,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           validation_data=(test_data, test_data),
#                           verbose=1,
#                           callbacks=[checkpointer, tensorboard]).history

# autoencoder = load_model('model.h5')
# plt.plot(history['loss'])
# plt.plot(history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()

# F1_gaussian = mgd_anomaly_detection(train_data, validation_data, test_data)

# all_data.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\all_data.csv', index=False)

# g = sns.pairplot(all_data, hue='ATT_FLAG', diag_kind='kde')
# plt.show()
# print(all_data)


# corr = all_data.corr()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0,len(all_data.columns),1)
# ax.set_xticks(ticks)
# plt.xticks(rotation=90)
# ax.set_yticks(ticks)
# ax.set_xticklabels(all_data.columns)
# ax.set_yticklabels(all_data.columns)
# plt.show()

# features = att_True.columns
# gs = gridspec.GridSpec(49, 49)
#
# for i, col in enumerate(features):
#     if np.count_nonzero(att_False[col].values) == 0:
#         print('Singular Matrix for ' + col)
#     else:
#         sns.distplot(att_False[col], color='g', label='No Attack Class', bins=100)
#
#     if np.count_nonzero(att_True[col].values) == 0:
#         print('Singular Matrix for ' + col)
#     else:
#         sns.distplot(att_True[col], color='r', label='Attack Class', bins=100)
#     plt.show()


# predictions = autoencoder.predict(test_data)
# mse = np.mean(np.power(test_data - predictions, 2), axis=1)
# error_df = pd.concat([mse, test_att_data], 1)
#
# error_df.columns = ['reconstruction_error', 'true_class']
#
# threshold = 2.9
#
# y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
# conf_matrix = confusion_matrix(error_df.true_class, y_pred)
# plt.figure(figsize=(12, 12))
# sns.heatmap(conf_matrix, annot=True, fmt="d")
# plt.title("Confusion matrix")
# plt.ylabel('True class')
# plt.xlabel('Predicted class')
# plt.show()
#
# print(f1_score(error_df.true_class, y_pred))
