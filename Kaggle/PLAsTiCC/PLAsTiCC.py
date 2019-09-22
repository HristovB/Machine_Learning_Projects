import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, LabelBinarizer, Normalizer
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
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

# matplotlib.use('GTK3Agg')

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


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

    # # Generation of statistical data from training_set (run once, then load)
    #
    # training_data = pd.read_csv("F:\Stuff\Data\\training_set.csv")
    # training_data = training_data.drop(['flux_err', 'mjd'], axis=1)
    #
    # mean_data = training_data[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).mean()
    # mn_dt = pd.DataFrame(np.reshape(mean_data.values, (7848, 6)), columns=['mn_u', 'mn_g', 'mn_r', 'mn_i', 'mn_z', 'mn_Y'])
    #
    # median_data = training_data[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).median()
    # md_dt = pd.DataFrame(np.reshape(median_data.values, (7848, 6)), columns=['md_u', 'md_g', 'md_r', 'md_i', 'md_z', 'md_Y'])
    #
    # max_data = training_data[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).max()
    # mx_dt = pd.DataFrame(np.reshape(max_data.values, (7848, 6)), columns=['mx_u', 'mx_g', 'mx_r', 'mx_i', 'mx_z', 'mx_Y'])
    #
    # min_data = training_data[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).min()
    # mi_dt = pd.DataFrame(np.reshape(min_data.values, (7848, 6)), columns=['mi_u', 'mi_g', 'mi_r', 'mi_i', 'mi_z', 'mi_Y'])
    #
    # std_data = training_data[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).std()
    # st_dt = pd.DataFrame(np.reshape(std_data.values, (7848, 6)), columns=['st_u', 'st_g', 'st_r', 'st_i', 'st_z', 'st_Y'])
    #
    # statistics = pd.concat([mn_dt, md_dt, mx_dt, mi_dt, st_dt], axis=1)
    #
    # scaler = StandardScaler()
    # scaled_vals = scaler.fit_transform(statistics)
    # statistics.loc[:, :] = scaled_vals
    #
    # avg_det = training_data[['object_id', 'detected', 'passband']].groupby(['object_id', 'passband']).mean()
    # avg_dt = pd.DataFrame(np.reshape(avg_det.values, (7848, 6)), columns=['det_avg_u', 'det_avg_g', 'det_avg_r', 'det_avg_i', 'det_avg_z', 'det_avg_Y'])
    #
    # statistics = pd.concat([statistics, avg_dt], axis=1)
    #
    # statistics.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Kaggle\PLAsTiCC\gen_statistics_training.csv', index=False)


    # # Setup full training data (run once, then load)
    #
    # training_metadata = pd.read_csv("F:\Stuff\Data\\training_set_metadata.csv")
    #
    # data_y = training_metadata.get(['target'])
    # # binarizer = LabelBinarizer()
    # # data_y = pd.DataFrame(binarizer.fit_transform(data_y.values), columns=['6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95'])
    #
    # training_metadata['hostgal_ph'] = (training_metadata['hostgal_photoz'] * training_metadata['hostgal_photoz_err'])
    # training_metadata = training_metadata.drop(['distmod', 'hostgal_specz', 'target', 'object_id', 'decl', 'hostgal_photoz', 'hostgal_photoz_err'], axis=1)
    # ddf = training_metadata.get(['ddf'])
    # training_metadata = training_metadata.drop(['ddf'], axis=1)
    #
    # scaler = StandardScaler()
    # scaled_vals = scaler.fit_transform(training_metadata)
    # training_metadata.loc[:, :] = scaled_vals
    # training_metadata = pd.concat([training_metadata, ddf], axis=1)
    #
    # statistics = pd.read_csv('gen_statistics_training.csv')
    #
    # data = pd.concat([training_metadata, statistics, data_y], axis=1)
    #
    # data.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Kaggle\PLAsTiCC\full_training_data_stats.csv', index=False)

    data = pd.read_csv('full_training_data_stats.csv')
    data_y = data['target']
    data_X = data.drop(['target'], axis=1)

    data_X, test_X, data_y, test_y = train_test_split(data_X, data_y, test_size=0.1, random_state=157, stratify=data_y.values)

    # data_y = data.get(['target'])
    # data_X = data.drop(['target'], axis=1)
    #
    # binarizer = LabelBinarizer()
    # data_y = pd.DataFrame(binarizer.fit_transform(data_y.values), columns=['6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95'])

    # Plot data
    #
    # data_615 = training_data.loc[training_data['object_id'] == 245853]
    #
    # plot_data = data_615.get(['mjd', 'flux', 'passband']).sort_values(by=['passband'])
    #
    # # plot_data = plot_data.loc[plot_data['mjd'] ]
    #
    # colors = ["#A025BE", "#25BE2C", "#DF2020", "#E89113", "#254CCF", "#000000"]
    # palette = sns.color_palette(colors)
    #
    # ax = sns.scatterplot(x='mjd', y='flux', hue='passband', data=plot_data, palette=palette)
    #
    # ax.set_title('Light Curve of 615')
    # ax.set_xlabel('MJD')
    # ax.set_ylabel('Flux')
    #
    # plt.show()
    # #

    # data_X = data_X.drop(['ra', 'gal_l', 'gal_b', 'mwebv', 'ddf', 'mn_z', 'mx_z', 'mi_z', 'st_z', 'md_z'], axis=1)
    # data_mutual_info = mutual_info_classif(X=data_X, y=data_y, random_state=157)
    #
    # plt.subplots(1, figsize=(26, 1))
    # sns.heatmap(data_mutual_info[:, np.newaxis].T, cmap='Blues', cbar=False, linewidths=1, annot=True)
    # plt.yticks([], [])
    # plt.gca().set_xticklabels(data_X.columns[0:], rotation=45, ha='right', fontsize=12)
    # plt.suptitle("mutual_info_classif)", fontsize=18, y=1.2)
    # plt.gcf().subplots_adjust(wspace=0.2)
    # plt.show()

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
    # plt.savefig('correlation.png')
    # plt.show()

    # One-Hot encoding the target classes

    # data_y = training_metadata.get(['target'])
    # binarizer = LabelBinarizer()
    # data_y = pd.DataFrame(binarizer.fit_transform(data_y.values), columns=['6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95'])

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

    # flux_data = pd.read_csv('flux_statistics.csv')
    #
    # scaler = StandardScaler()
    # scaled_vals = scaler.fit_transform(flux_data)
    # flux_data.loc[:, :] = scaled_vals
    #
    # data_X = training_metadata.drop(['object_id', 'target', 'distmod'], axis=1)
    # scaler = StandardScaler()
    # scaled_vals = scaler.fit_transform(data_X)
    # data_X.loc[:, :] = scaled_vals
    #
    # data_X = pd.concat([data_X, flux_data], axis=1)

    # cv = StratifiedKFold(n_splits=10, random_state=157)

    # for n_features in range(5, data_X.shape[1]):
    #     pca = PCA(n_components=n_features, whiten=True, random_state=157)
    #     pca.fit(data_X)
    #     pca.get_covariance()
    #     data_X_pca = pd.DataFrame(pca.transform(data_X))
    #

    # n_estimators = [200, 300, 500, 750, 1000, 1200]
    # # learning_rate = [0.01, 0.03, 0.1, 0.3, 1, 3]
    # # max_depth = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
    # max_features = np.arange(5, data_X.shape[1], 5)
    # max_features = np.append(max_features, (data_X.shape[1]))
    # # min_samples_leaf = [1, 3, 5, 7, 10]
    # # min_samples_split = [2, 5, 7, 10]
    #
    # hyperF = dict(n_estimators=n_estimators, max_features=max_features)
    # clf = ExtraTreesClassifier(random_state=157)
    #
    # gridF = GridSearchCV(clf, hyperF, cv=10, verbose=1, n_jobs=-1)
    # bestF = gridF.fit(data_X, data_y.values.flatten())
    # print(gridF.cv_results_)
    # results = pd.DataFrame(gridF.cv_results_)
    # results.to_csv(r'results_et_3.csv', index=False)

    # binarizer = LabelBinarizer()
    # test_y = pd.DataFrame(binarizer.fit_transform(test_y.values), columns=['6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95'])



    # clf = AdaBoostClassifier(random_state=157, base_estimator=RandomForestClassifier(n_estimators=750, random_state=157, n_jobs=-1))
    #
    # clf.fit(data_X, data_y)
    #

    # from keras.models import Sequential, load_model
    # from keras.layers import Dense, BatchNormalization, Dropout
    # from keras.callbacks import ModelCheckpoint, TensorBoard
    # from keras import regularizers
    # import tensorflow as tf
    #
    # set_seed()
    #
    # neural_net = load_model('model.h5')
    #
    # y_predicted = neural_net.predict_proba(test_X)
    #
    # score = multi_weighted_logloss(test_y, y_predicted)
    #
    # print(score)

    # y_predicted = clf.predict(test_X)
    # # y_predicted = pd.DataFrame(binarizer.fit_transform(y_predicted))
    #
    # cf_matrix = confusion_matrix(test_y, y_predicted)
    #
    # labels = ['6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95']
    # ax = sns.heatmap(cf_matrix, annot=True, xticklabels=labels, yticklabels=labels, cbar=False, cmap='Blues')
    # ax.set_title('Confusion Matrix - Extra Trees')
    # ax.set_xlabel('Predicted label')
    # ax.set_ylabel('True label')
    # plt.show()

# ---------------------------------------------------- NN ------------------------------------------------------------ #

    from keras.models import Sequential, load_model
    from keras.layers import Dense, BatchNormalization, Dropout
    from keras.callbacks import ModelCheckpoint, TensorBoard
    from keras.initializers import he_uniform, glorot_uniform
    from keras import regularizers
    import tensorflow as tf

    set_seed()

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.1, random_state=157, stratify=data_y.values)

    binarizer = LabelBinarizer()
    y_train = pd.DataFrame(binarizer.fit_transform(y_train.values), columns=['6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95'])
    y_test = pd.DataFrame(binarizer.fit_transform(y_test.values), columns=['6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95'])

    nb_epoch = 600
    batch_size = 256
    input_dim = X_train.shape[1]
    output_dim = 14
    layer_size = output_dim*16
    init_relu = he_uniform(seed=157)
    init_tanh = glorot_uniform(seed=157)

    nn = Sequential()

    nn.add(Dense(layer_size, input_dim=input_dim, kernel_initializer=init_tanh, activation='tanh'))
    nn.add(BatchNormalization())
    nn.add(Dropout(0.25))

    nn.add(Dense(int(layer_size / 2), kernel_initializer=init_relu, activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Dropout(0.25))

    nn.add(Dense(int(layer_size / 4), kernel_initializer=init_tanh, activation='tanh'))
    nn.add(BatchNormalization())
    nn.add(Dropout(0.25))

    nn.add(Dense(int(layer_size / 8), kernel_initializer=init_relu, activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Dropout(0.125))

    nn.add(Dense(output_dim, kernel_initializer=init_tanh, activation='softmax'))

    nn.compile(optimizer='adam',
               loss='mean_squared_logarithmic_error',
               metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath="model.h5",
                                   verbose=0,
                                   save_best_only=True)

    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    history = nn.fit(x=X_train, y=y_train,
                     validation_data=[X_test, y_test],
                     epochs=nb_epoch,
                     batch_size=batch_size,
                     shuffle=True,
                     verbose=1,
                     callbacks=[checkpointer, tensorboard]).history

    y_predicted = nn.predict_proba(test_X, batch_size=batch_size)

    score = multi_weighted_logloss(test_y, y_predicted)
    print(score)

    # neural_net = load_model('model.h5')
    # data_y_pred = neural_net.predict_proba(data_X, batch_size=256)
    # score = multi_weighted_logloss(data_y, data_y_pred)
    # print(score)

    # # Extract statistical features for test set (run once, then load)
    # chunksize = 50000000
    # for chunk in pd.read_csv('F:\Stuff\Data\\test_set.csv', chunksize=chunksize):
    #     print(chunk)
    #
    #     # Generation of statistical data from training_set (run once, then load)
    #
    #     chunk = chunk.drop(['flux_err', 'mjd'], axis=1)
    #     # count = len(chunk['object_id'].unique())
    #
    #     mean_data = chunk[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).mean()
    #
    #     median_data = chunk[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).median()
    #
    #     max_data = chunk[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).max()
    #
    #     min_data = chunk[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).min()
    #
    #     std_data = chunk[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).std()
    #
    #     statistics = pd.concat([mean_data, median_data, max_data, min_data, std_data], axis=1)
    #
    #     avg_det = chunk[['object_id', 'detected', 'passband']].groupby(['object_id', 'passband']).mean()
    #
    #     statistics = pd.concat([statistics, avg_det], axis=1)
    #
    #     statistics.to_csv(r'F:\Stuff\Data\gen_statistics_test.csv', index=False, mode='a')
    #
    #     print('Chunk done!')
    # print('Done!')

    # test_stats = pd.read_csv('F:\Stuff\Data\\gen_statistics_test.csv')
    # test_stats = test_stats.convert_objects(convert_numeric=True)
    #
    # print(test_stats)
    #
    # length = int(test_stats.shape[0] / 6)
    #
    # mn_dt = pd.DataFrame(np.reshape(test_stats.iloc[:, 0].values, (length, 6)), columns=['mn_u', 'mn_g', 'mn_r', 'mn_i', 'mn_z', 'mn_Y'], dtype='float32')
    #
    # md_dt = pd.DataFrame(np.reshape(test_stats.iloc[:, 1].values, (length, 6)), columns=['md_u', 'md_g', 'md_r', 'md_i', 'md_z', 'md_Y'], dtype='float32')
    #
    # mx_dt = pd.DataFrame(np.reshape(test_stats.iloc[:, 2].values, (length, 6)), columns=['mx_u', 'mx_g', 'mx_r', 'mx_i', 'mx_z', 'mx_Y'], dtype='float32')
    #
    # mi_dt = pd.DataFrame(np.reshape(test_stats.iloc[:, 3].values, (length, 6)), columns=['mi_u', 'mi_g', 'mi_r', 'mi_i', 'mi_z', 'mi_Y'], dtype='float32')
    #
    # st_dt = pd.DataFrame(np.reshape(test_stats.iloc[:, 4].values, (length, 6)), columns=['st_u', 'st_g', 'st_r', 'st_i', 'st_z', 'st_Y'], dtype='float32')
    #
    # statistics = pd.concat([mn_dt, md_dt, mx_dt, mi_dt, st_dt], axis=1)
    #
    # scaler = StandardScaler()
    # scaled_vals = scaler.fit_transform(statistics)
    # statistics.loc[:, :] = scaled_vals
    #
    # avg_dt = pd.DataFrame(np.reshape(test_stats.iloc[:, 5].values, (length, 6)), columns=['det_avg_u', 'det_avg_g', 'det_avg_r', 'det_avg_i', 'det_avg_z', 'det_avg_Y'], dtype='float32')
    #
    # statistics = pd.concat([statistics, avg_dt], axis=1,)
    #
    # statistics.to_csv(r'F:\Stuff\Data\\full_test_statistics.csv', index=False)

    # test_metadata = pd.read_csv('F:\Stuff\Data\\test_set_metadata.csv')
    # object_id = test_metadata.get(['object_id'])
    # print(object_id)
    # #
    # test_metadata['hostgal_ph'] = (test_metadata['hostgal_photoz'] * test_metadata['hostgal_photoz_err'])
    # test_metadata = test_metadata.drop(['distmod', 'hostgal_specz', 'object_id', 'decl', 'hostgal_photoz', 'hostgal_photoz_err'], axis=1)
    # ddf = test_metadata.get(['ddf'])
    # test_metadata = test_metadata.drop(['ddf'], axis=1)
    # print('Edited test metadata')
    #
    # scaler = StandardScaler()
    # scaled_vals = scaler.fit_transform(test_metadata)
    # test_metadata.loc[:, :] = scaled_vals
    # test_metadata = pd.concat([test_metadata, ddf], axis=1)
    # print('Scaled test metadata')
    #
    # statistics = pd.read_csv('F:\Stuff\Data\\full_test_statistics.csv')
    # print('Read test statistic data')
    #
    # data = pd.concat([test_metadata, statistics.iloc[:-10, :]], axis=1)
    # print('Saving...')
    #
    # data.to_csv(r'F:\Stuff\Data\\full_test_data.csv', index=False)

    # test_data = pd.read_csv('F:\Stuff\Data\\full_test_data.csv')
    # print('Done reading')
    #
    # data_y = pd.DataFrame(neural_net.predict_proba(test_data, batch_size=256))
    # data_y = pd.concat([object_id, data_y], axis=1)
    # data_y.to_csv(r'F:\Stuff\Data\\predictions.csv', index=False)
