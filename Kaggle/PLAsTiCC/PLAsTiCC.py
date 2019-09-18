import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, LabelBinarizer, Normalizer
from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import IsolationForest, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

matplotlib.use('GTK3Agg')


def calculate_f1(y_true, y_pred=None, p=None, epsilon=None):
    y_true = y_true.values.flatten().astype(int)

    if epsilon != None:
        y_pred = (p < epsilon).astype(int)

    tp = np.sum((y_pred == 1).astype(int) & (y_true == 1).astype(int))
    tn = np.sum((y_pred == 0).astype(int) & (y_true == 0).astype(int))
    fp = np.sum((y_pred == 1).astype(int) & (y_true == 0).astype(int))
    fn = np.sum((y_pred == 0).astype(int) & (y_true == 1).astype(int))

    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)

    f1 = (TPR + TNR) / 2

    return f1, tn, fp, fn, tp


def multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


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


if __name__ == '__main__':

    training_data = pd.read_csv("/home/bhristov/Documents/Programming/PyCharm/PLAsTiCC/training_set.csv")
    training_metadata = pd.read_csv("/home/bhristov/Documents/Programming/PyCharm/PLAsTiCC/training_set_metadata.csv")

    # One-Hot encoding the target classes

    data_y = training_metadata.get(['target'])
    binarizer = LabelBinarizer()
    data_y = pd.DataFrame(binarizer.fit_transform(data_y.values), columns=['6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95'])

    # # Plot data
    #
    # data_615 = training_data.loc[training_data['object_id'] == 615]
    #
    # plot_data = data_615.get(['mjd', 'flux', 'passband']).sort_values(by=['passband'])
    #
    # # plot_data = plot_data.loc[plot_data['mjd'] ]
    #
    # colors = ["#A025BE", "#25BE2C", "#DF2020", "#E89113", "#254CCF", "#000000"]
    # palette = sns.color_palette(colors)
    #
    # ax = sns.lineplot(x='mjd', y='flux', hue='passband', data=plot_data, palette=palette)
    #
    # ax.set_title('Light Curve of 615')
    # ax.set_xlabel('MJD')
    # ax.set_ylabel('Flux')
    #
    # plt.show()
    # #

    passband = training_data['passband']
    binarizer = LabelBinarizer()
    passband = pd.DataFrame(binarizer.fit_transform(passband.values), columns=['u', 'g', 'r', 'i', 'z', 'Y'])

    training_data = training_data.drop(['passband'], axis=1)
    training_data = pd.concat([training_data, passband], axis=1)

    flux_data = pd.read_csv('flux_statistics.csv')

    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(flux_data)
    flux_data.loc[:, :] = scaled_vals

    data_X = training_metadata.drop(['object_id', 'target', 'distmod'], axis=1)
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(data_X)
    data_X.loc[:, :] = scaled_vals

    data_X = pd.concat([data_X, flux_data], axis=1)


    # # Feature correlation plot
    #
    # corr = data_X.corr()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    # fig.colorbar(cax)
    # ticks = np.arange(0, len(data_X.columns), 1)
    # ax.set_xticks(ticks)
    # plt.xticks(rotation=90)
    # ax.set_yticks(ticks)
    # ax.set_xticklabels(data_X.columns)
    # ax.set_yticklabels(data_X.columns)
    # plt.show()
    # #

    # # Scaling

    # scaler = StandardScaler()
    # to_scale = training_data.get(['flux', 'flux_err'])
    # scaled_vals = pd.DataFrame(scaler.fit_transform(to_scale), columns=['flux', 'flux_err'])
    # training_data[['flux', 'flux_err']] = scaled_vals

    # scaler = MinMaxScaler()
    # to_scale = training_data.get(['object_id', 'mjd'])
    # scaled_vals = pd.DataFrame(scaler.fit_transform(to_scale), columns=['object_id', 'mjd'])
    # training_data[['object_id', 'mjd']] = scaled_vals
    # #

    # # Calculate statistical features of flux for each individual object for each passband (one-time run, open csv after)
    # new_features = pd.DataFrame([])
    #
    # for obj in pd.unique(training_data['object_id'].values):
    #
    #     mean_u = np.mean(training_data.loc[(training_data['object_id'] == obj) & (training_data['u'] == 1), 'flux'])
    #     mean_g = np.mean(training_data.loc[(training_data['object_id'] == obj) & (training_data['g'] == 1), 'flux'])
    #     mean_r = np.mean(training_data.loc[(training_data['object_id'] == obj) & (training_data['r'] == 1), 'flux'])
    #     mean_i = np.mean(training_data.loc[(training_data['object_id'] == obj) & (training_data['i'] == 1), 'flux'])
    #     mean_z = np.mean(training_data.loc[(training_data['object_id'] == obj) & (training_data['z'] == 1), 'flux'])
    #     mean_Y = np.mean(training_data.loc[(training_data['object_id'] == obj) & (training_data['Y'] == 1), 'flux'])
    #
    #     maximum_u = np.max(training_data.loc[(training_data['object_id'] == obj) & (training_data['u'] == 1), 'flux'])
    #     maximum_g = np.max(training_data.loc[(training_data['object_id'] == obj) & (training_data['g'] == 1), 'flux'])
    #     maximum_r = np.max(training_data.loc[(training_data['object_id'] == obj) & (training_data['r'] == 1), 'flux'])
    #     maximum_i = np.max(training_data.loc[(training_data['object_id'] == obj) & (training_data['i'] == 1), 'flux'])
    #     maximum_z = np.max(training_data.loc[(training_data['object_id'] == obj) & (training_data['z'] == 1), 'flux'])
    #     maximum_Y = np.max(training_data.loc[(training_data['object_id'] == obj) & (training_data['Y'] == 1), 'flux'])
    #
    #     minimum_u = np.min(training_data.loc[(training_data['object_id'] == obj) & (training_data['u'] == 1), 'flux'])
    #     minimum_g = np.min(training_data.loc[(training_data['object_id'] == obj) & (training_data['g'] == 1), 'flux'])
    #     minimum_r = np.min(training_data.loc[(training_data['object_id'] == obj) & (training_data['r'] == 1), 'flux'])
    #     minimum_i = np.min(training_data.loc[(training_data['object_id'] == obj) & (training_data['i'] == 1), 'flux'])
    #     minimum_z = np.min(training_data.loc[(training_data['object_id'] == obj) & (training_data['z'] == 1), 'flux'])
    #     minimum_Y = np.min(training_data.loc[(training_data['object_id'] == obj) & (training_data['Y'] == 1), 'flux'])
    #
    #     std_u = np.std(training_data.loc[(training_data['object_id'] == obj) & (training_data['u'] == 1), 'flux'])
    #     std_g = np.std(training_data.loc[(training_data['object_id'] == obj) & (training_data['g'] == 1), 'flux'])
    #     std_r = np.std(training_data.loc[(training_data['object_id'] == obj) & (training_data['r'] == 1), 'flux'])
    #     std_i = np.std(training_data.loc[(training_data['object_id'] == obj) & (training_data['i'] == 1), 'flux'])
    #     std_z = np.std(training_data.loc[(training_data['object_id'] == obj) & (training_data['z'] == 1), 'flux'])
    #     std_Y = np.std(training_data.loc[(training_data['object_id'] == obj) & (training_data['Y'] == 1), 'flux'])
    #
    #     new_features = new_features.append([[mean_u, mean_g, mean_r, mean_i, mean_z, mean_Y,
    #                                          maximum_u, maximum_g, maximum_r, maximum_i, maximum_z, maximum_Y,
    #                                          minimum_u, minimum_g, minimum_r, minimum_i, minimum_z, minimum_Y,
    #                                          std_u, std_g, std_r, std_i, std_z, std_Y,]])
    #
    # new_features.to_csv(r'/home/bhristov/Documents/Programming/PyCharm/Machine_Learning/Kaggle/PLAsTiCC/flux_statistics.csv', index=False)
    # print('Done!')
    # #


    # cv = StratifiedKFold(n_splits=10, random_state=157)

    # best_loss = np.inf
    # best_features = 0
    # best_trees = 0
    # loss_sum = 0
    # loss_avg = 0
    #
    # # for n_features in range(5, data_X.shape[1]):
    # #     pca = PCA(n_components=n_features, whiten=True, random_state=157)
    # #     pca.fit(data_X)
    # #     pca.get_covariance()
    # #     data_X_pca = pd.DataFrame(pca.transform(data_X))
    #
    # for n_trees in range(1, 201, 5):
    #     # clf_grad = KNeighborsClassifier(random_state=157, n_neighbors=5)
    #     clf = XGBClassifier(random_state=157, n_estimators=n_trees, eta=0.1)
    #
    #     for train_index, test_index in cv.split(data_X, data_y):
    #         X_train, X_test = training_data.iloc[train_index, :], training_data.iloc[test_index, :]
    #         y_train, y_test = data_y.iloc[train_index, :], data_y.iloc[test_index, :]
    #
    #         clf.fit(X_train, y_train.values.flatten())
    #         y_proba = clf.predict_proba(X_test)
    #
    #         loss_sum += multi_weighted_logloss(y_test, y_proba)
    #
    #     loss_avg = loss_sum/10
    #
    #     if loss_avg < best_loss:
    #         best_loss = loss_avg
    #         best_trees = n_trees
    #         # best_features = n_features
    #
    #     print(loss_avg, n_trees)
    #     print(best_loss, best_trees)
    #     loss_avg = 0
    #     loss_sum = 0
    #
    # print(best_loss, best_trees)

# ---------------------------------------------------- NN ------------------------------------------------------------ #

    # from keras.models import Sequential, load_model
    # from keras.layers import Dense, BatchNormalization, Dropout
    # from keras.callbacks import ModelCheckpoint, TensorBoard
    # from keras import regularizers
    # import tensorflow as tf
    #
    # set_seed()
    #
    # cv = KFold(n_splits=10, random_state=157)

    # tensorboard = TensorBoard(log_dir='./logs',
    #                           histogram_freq=0,
    #                           write_graph=True,
    #                           write_images=True)
    #
    # best_score = np.inf
    #
    # for train_index, test_index in cv.split(data_X, data_y):
    #     checkpointer = ModelCheckpoint(filepath="model.h5",
    #                                    verbose=0,
    #                                    save_best_only=True)
    #
    #     X_train, X_test = data_X.iloc[train_index, :], data_X.iloc[test_index, :]
    #     y_train, y_test = data_y.iloc[train_index, :], data_y.iloc[test_index, :]
    #
    #     input_dim = X_train.shape[1]
    #     layer_size = 14*16
    #     nn = Sequential()
    #
    #     nn.add(Dense(layer_size, input_dim=input_dim, activation='tanh'))
    #     nn.add(BatchNormalization())
    #     nn.add(Dropout(0.25))
    #
    #     nn.add(Dense(int(layer_size/2), activation='relu'))
    #     nn.add(BatchNormalization())
    #     nn.add(Dropout(0.25))
    #
    #     nn.add(Dense(int(layer_size/4), activation='relu'))
    #     nn.add(BatchNormalization())
    #     nn.add(Dropout(0.25))
    #
    #     nn.add(Dense(int(layer_size/8), activation='tanh'))
    #     nn.add(BatchNormalization())
    #     nn.add(Dropout(0.125))
    #
    #     nn.add(Dense(14, activation='softmax'))
    #
    #     nb_epoch = 300
    #     batch_size = 128
    #
    #     nn.compile(optimizer='adam',
    #                loss='mean_squared_error',
    #                metrics=['accuracy'])
    #
    #     history = nn.fit(x=X_train, y=y_train,
    #                      validation_data=[X_test, y_test],
    #                      epochs=nb_epoch,
    #                      batch_size=batch_size,
    #                      shuffle=True,
    #                      verbose=1,
    #                      callbacks=[checkpointer, tensorboard]).history
    #
    #     logloss_score = multi_weighted_logloss(y_test, nn.predict_proba(X_test))
    #     print(logloss_score)
    #
    #     best_score = min(logloss_score, best_score)
    #
    # print(best_score)

    # neural_net = load_model('model.h5')
    # data_y_pred = neural_net.predict_proba(data_X, batch_size=128)
    # print(data_y_pred)

    chunksize = 5000000
    for chunk in pd.read_csv('/home/bhristov/Documents/Programming/PyCharm/PLAsTiCC/test_set.csv', chunksize=chunksize):
        print(chunk)
        passband = chunk['passband']
        binarizer = LabelBinarizer()
        passband = pd.DataFrame(binarizer.fit_transform(passband.values), columns=['u', 'g', 'r', 'i', 'z', 'Y'])

        chunk = chunk.drop(['passband'], axis=1)
        chunk = pd.concat([chunk, passband], axis=1)

        # Calculate statistical features of flux for each individual object for each passband (one-time run, open csv after)
        new_features = pd.DataFrame([])

        for obj in pd.unique(chunk['object_id'].values):

            mean_u = np.mean(chunk.loc[(chunk['object_id'] == obj) & (chunk['u'] == 1), 'flux'])
            mean_g = np.mean(chunk.loc[(chunk['object_id'] == obj) & (chunk['g'] == 1), 'flux'])
            mean_r = np.mean(chunk.loc[(chunk['object_id'] == obj) & (chunk['r'] == 1), 'flux'])
            mean_i = np.mean(chunk.loc[(chunk['object_id'] == obj) & (chunk['i'] == 1), 'flux'])
            mean_z = np.mean(chunk.loc[(chunk['object_id'] == obj) & (chunk['z'] == 1), 'flux'])
            mean_Y = np.mean(chunk.loc[(chunk['object_id'] == obj) & (chunk['Y'] == 1), 'flux'])

            maximum_u = np.max(chunk.loc[(chunk['object_id'] == obj) & (chunk['u'] == 1), 'flux'])
            maximum_g = np.max(chunk.loc[(chunk['object_id'] == obj) & (chunk['g'] == 1), 'flux'])
            maximum_r = np.max(chunk.loc[(chunk['object_id'] == obj) & (chunk['r'] == 1), 'flux'])
            maximum_i = np.max(chunk.loc[(chunk['object_id'] == obj) & (chunk['i'] == 1), 'flux'])
            maximum_z = np.max(chunk.loc[(chunk['object_id'] == obj) & (chunk['z'] == 1), 'flux'])
            maximum_Y = np.max(chunk.loc[(chunk['object_id'] == obj) & (chunk['Y'] == 1), 'flux'])

            minimum_u = np.min(chunk.loc[(chunk['object_id'] == obj) & (chunk['u'] == 1), 'flux'])
            minimum_g = np.min(chunk.loc[(chunk['object_id'] == obj) & (chunk['g'] == 1), 'flux'])
            minimum_r = np.min(chunk.loc[(chunk['object_id'] == obj) & (chunk['r'] == 1), 'flux'])
            minimum_i = np.min(chunk.loc[(chunk['object_id'] == obj) & (chunk['i'] == 1), 'flux'])
            minimum_z = np.min(chunk.loc[(chunk['object_id'] == obj) & (chunk['z'] == 1), 'flux'])
            minimum_Y = np.min(chunk.loc[(chunk['object_id'] == obj) & (chunk['Y'] == 1), 'flux'])

            std_u = np.std(chunk.loc[(chunk['object_id'] == obj) & (chunk['u'] == 1), 'flux'])
            std_g = np.std(chunk.loc[(chunk['object_id'] == obj) & (chunk['g'] == 1), 'flux'])
            std_r = np.std(chunk.loc[(chunk['object_id'] == obj) & (chunk['r'] == 1), 'flux'])
            std_i = np.std(chunk.loc[(chunk['object_id'] == obj) & (chunk['i'] == 1), 'flux'])
            std_z = np.std(chunk.loc[(chunk['object_id'] == obj) & (chunk['z'] == 1), 'flux'])
            std_Y = np.std(chunk.loc[(chunk['object_id'] == obj) & (chunk['Y'] == 1), 'flux'])

            new_features = new_features.append([[mean_u, mean_g, mean_r, mean_i, mean_z, mean_Y,
                                                 maximum_u, maximum_g, maximum_r, maximum_i, maximum_z, maximum_Y,
                                                 minimum_u, minimum_g, minimum_r, minimum_i, minimum_z, minimum_Y,
                                                 std_u, std_g, std_r, std_i, std_z, std_Y]])

        new_features.to_csv(r'/home/bhristov/Documents/Programming/PyCharm/Machine_Learning/Kaggle/PLAsTiCC/flux_statistics_test.csv', index=False, mode='a')
        print('Chunk done!')
    print('Done!')

