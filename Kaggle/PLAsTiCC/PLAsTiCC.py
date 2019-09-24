import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier


# Uncomment if running on laptop:
# import matplotlib
# matplotlib.use('GTK3Agg')

# Setup pandas and numpy printing options
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=np.inf)


def set_seed():

    import os
    import random
    import tensorflow as tf

    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value = 157

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value

    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value

    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value

    tf.compat.v1.set_random_seed(157)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as k

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                            inter_op_parallelism_threads=1)

    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    k.set_session(sess)


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


def generate_statistics(data, method='old'):

    # ----- Generation of statistical data from training_set (run once, then load) ----- #

    if method != 'old' and method != 'new':
        raise ValueError('Wrong input for method! Acceptable inputs: \'old\', \'new\'')

    num_rows = 7848  # number of objects in dataset
    num_passbands = 6  # number of passband filters

    # Generate statistical data for flux time-series
    mean_data = data[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).mean()
    mn_dt = pd.DataFrame(np.reshape(mean_data.values, (num_rows, num_passbands)), columns=['mn_u', 'mn_g', 'mn_r', 'mn_i', 'mn_z', 'mn_Y'])

    median_data = data[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).median()
    md_dt = pd.DataFrame(np.reshape(median_data.values, (num_rows, num_passbands)), columns=['md_u', 'md_g', 'md_r', 'md_i', 'md_z', 'md_Y'])

    max_data = data[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).max()
    mx_dt = pd.DataFrame(np.reshape(max_data.values, (num_rows, num_passbands)), columns=['mx_u', 'mx_g', 'mx_r', 'mx_i', 'mx_z', 'mx_Y'])

    min_data = data[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).min()
    mi_dt = pd.DataFrame(np.reshape(min_data.values, (num_rows, num_passbands)), columns=['mi_u', 'mi_g', 'mi_r', 'mi_i', 'mi_z', 'mi_Y'])

    std_data = data[['object_id', 'flux', 'passband']].groupby(['object_id', 'passband']).std()
    st_dt = pd.DataFrame(np.reshape(std_data.values, (num_rows, num_passbands)), columns=['st_u', 'st_g', 'st_r', 'st_i', 'st_z', 'st_Y'])

    statistics = pd.concat([mn_dt, md_dt, mx_dt, mi_dt, st_dt], axis=1)

    # Standardize values of all columns
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(statistics)
    statistics.loc[:, :] = scaled_vals

    # Generate mean value of 'detected' column
    avg_det = data[['object_id', 'detected', 'passband']].groupby(['object_id', 'passband']).mean()
    avg_dt = pd.DataFrame(np.reshape(avg_det.values, (7848, 6)), columns=['det_avg_u', 'det_avg_g', 'det_avg_r', 'det_avg_i', 'det_avg_z', 'det_avg_Y'])

    statistics = pd.concat([statistics, avg_dt], axis=1)

    if method == 'new':
        # Generate super - feature for determining cyclicity of object
        tmp_df = data[data['detected'] == 1]
        mjd_min = tmp_df[['object_id', 'mjd']].groupby(['object_id']).min()
        mjd_max = tmp_df[['object_id', 'mjd']].groupby(['object_id']).max()
        mjd_diff = mjd_max - mjd_min
        mjd_diff = pd.DataFrame(mjd_diff.values, columns=['mjd_diff'])  # needs to be appended later

        # Generate two new features incorporating the flux error
        data['flux_ratio_sq'] = (data['flux'] / data['flux_err']) ** 2
        data['flux_by_flux_ratio_sq'] = data['flux'] * data['flux_ratio_sq']

        rt_sq = data[['object_id', 'flux_ratio_sq', 'passband']].groupby(['object_id', 'passband']).sum()
        rtsq_dt = pd.DataFrame(np.reshape(rt_sq.values, (num_rows, num_passbands)), columns=['rtsq_u', 'rtsq_g', 'rtsq_r', 'rtsq_i', 'rtsq_z', 'rtsq_Y'])

        fl_by_rt = data[['object_id', 'flux_by_flux_ratio_sq', 'passband']].groupby(['object_id', 'passband']).sum()
        flrt_dt = pd.DataFrame(np.reshape(fl_by_rt.values, (num_rows, num_passbands)), columns=['flrt_u', 'flrt_g', 'flrt_r', 'flrt_i', 'flrt_z', 'flrt_Y'])

        new_statistics = pd.concat([rtsq_dt, flrt_dt, mjd_diff], axis=1)

        # Standardize values of all columns
        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(new_statistics)
        new_statistics.loc[:, :] = scaled_vals

        statistics = pd.concat([statistics, new_statistics], axis=1)

    # Save data
    statistics.to_csv(r'gen_statistics_training.csv', index=False)

    return statistics


def setup_data(data, metadata, method='old'):

    # ----- Final setup of training data (run once, then load) ----- #

    # Extract output labels into separate dataframe
    labels = metadata.get(['target'])

    # Generate new feature
    metadata['hostgal_ph_ratio_sq'] = (metadata['hostgal_photoz'] / metadata['hostgal_photoz_err']) ** 2
    metadata['hostgal_ph_ratio_sq'] = metadata['hostgal_ph_ratio_sq'].fillna(value=0)

    # Drop unnecessary features
    metadata = metadata.drop(['distmod', 'hostgal_specz', 'target', 'object_id', 'decl', 'hostgal_photoz', 'hostgal_photoz_err'], axis=1)

    # Seperate 'ddf' binary feature into separate dataframe before standardization
    ddf = metadata.get(['ddf'])
    metadata = metadata.drop(['ddf'], axis=1)

    # Standardize values of all columns
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(metadata)
    metadata.loc[:, :] = scaled_vals

    # Append 'ddf' feature
    metadata = pd.concat([metadata, ddf], axis=1)

    # Generate new statistical features
    statistics = generate_statistics(data=data, method=method)

    # Append all dataframes
    final_data = pd.concat([metadata, statistics, labels], axis=1)

    # Save data
    final_data.to_csv(r'full_training_data.csv', index=False)

    return final_data


def correlation_plot(X, y, method='feature'):

    # ----- Correlation plot ----- #

    if method == 'feature':

        # Plot feature correlation
        corr = X.corr()
        fig = plt.figure()
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

        plt.subplots(1, figsize=(26, 1))
        sns.heatmap(data_mutual_info[:, np.newaxis].T, cmap='Blues', cbar=False, linewidths=1, annot=True)
        plt.yticks([], [])
        plt.gca().set_xticklabels(data_X.columns[0:], rotation=45, ha='right', fontsize=12)
        plt.suptitle("mutual_info_classif)", fontsize=18, y=1.2)
        plt.gcf().subplots_adjust(wspace=0.2)
        plt.savefig('output_correlation.png')
        plt.show()

    elif method != 'feature' and method != 'label':
        raise ValueError('Wrong input for method! Acceptable inputs: \'feature\', \'label\'')


def light_curves_plot(data, object_id=615):

    # ----- Light curve plot (use original training dataset) ----- #

    possible_id = data['object_id'].unique()

    if object_id not in possible_id:
        raise ValueError('Wrong input for object_id! Acceptable inputs:\n' + str(possible_id))

    object_data = data.loc[data['object_id'] == object_id]

    passband = object_data['passband'].values
    encoder = {0: 'u', 1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'Y'}
    passband = pd.DataFrame([encoder[element] for element in passband], columns=['passband'])

    plot_data = object_data.get(['mjd', 'flux']).reset_index()
    plot_data = plot_data.drop(['index'], axis=1)
    plot_data = pd.concat([plot_data, passband], axis=1)

    palette = {'u': "#A025BE", 'g': "#25BE2C", 'r': "#DF2020", 'i': "#E89113", 'z': "#254CCF", 'Y': "#000000"}

    ax = sns.scatterplot(x='mjd', y='flux', hue='passband', hue_order=['u', 'g', 'r', 'i', 'z', 'Y'], data=plot_data, palette=palette)
    ax.set_title('Light Curve of ' + str(object_id))
    ax.set_xlabel('MJD')
    ax.set_ylabel('Flux')
    plt.show()


def grid_search(X, y):

    # ----- Grid search with 10 fold CV for model hyperparameters ----- #

    # Seed for random generator
    seed = 157

    # Hyperparameters to tune
    n_estimators = [200, 300, 500, 750, 1000, 1200]
    # learning_rate = [0.01, 0.03, 0.1, 0.3, 1, 3]
    # max_depth = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
    max_features = np.arange(5, data_X.shape[1], 5)
    max_features = np.append(max_features, (data_X.shape[1]))
    # min_samples_leaf = [1, 3, 5, 7, 10]
    # min_samples_split = [2, 5, 7, 10]

    # Python dictionary with hyperparameters
    param_dict = dict(n_estimators=n_estimators, max_features=max_features)

    # Setup classifier (don't forget to set random seed if possible!)
    clf = ExtraTreesClassifier(random_state=seed)

    # File naming convention
    clf_name = 'et'  # short name of classifier
    iteration = '3'  # number of iterations with that classifier

    # Setup grid search algorithm from sklearn
    grid = GridSearchCV(estimator=clf, param_grid=param_dict, cv=10, verbose=1, n_jobs=-1)

    # Run grid search algorithm
    grid.fit(X, y.values.flatten())

    # Get results for all tuned hyperparameters
    results = pd.DataFrame(grid.cv_results_)

    # Save data
    results.to_csv(r'results_' + clf_name + '_' + iteration + '.csv', index=False)


if __name__ == '__main__':

    # # Run once, then load from full_training_data.csv
    # training_data = pd.read_csv("F:\Stuff\Data\\training_set.csv")
    # training_metadata = pd.read_csv("F:\Stuff\Data\\training_set_metadata.csv")
    # all_data = setup_data(training_data, training_metadata, method='new')

    # Only run if data setup is done!
    all_data = pd.read_csv('full_training_data.csv')

    # Separate feature data and labels
    data_y = all_data['target']
    data_X = all_data.drop(['target'], axis=1)

    # correlation_plot(X=data_X, y=data_y)  # Plot for feature/label correlation

    # light_curves_plot(training_data, object_id=713)  # Scatterplot for light curve data of individual objects (use original training dataset)

    # Split off 10% of the data for testing (use stratify to get proportional amount of labels in both sets - useful for imbalanced datasets)
    data_X, test_X, data_y, test_y = train_test_split(data_X, data_y, test_size=0.1, random_state=157, stratify=data_y.values)

    # grid_search(data_X, data_y)  # Grid search for optimal hyperparameters (look at function to change classifier and parameters)

    # Training and testing of ensemble algorithms
    clf = ExtraTreesClassifier(n_estimators=1200, max_features=data_X.shape[1], n_jobs=-1, random_state=157)
    clf.fit(data_X, data_y)

    # Predict test labels
    y_predicted = clf.predict_proba(test_X)

    # Calculate loss
    score = multi_weighted_logloss(test_y, y_predicted)
    print(score)

    # Confusion matrix plot
    y_predicted = clf.predict(test_X)

    cf_matrix = confusion_matrix(test_y, y_predicted)

    labels = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    ax = sns.heatmap(cf_matrix, annot=True, xticklabels=labels, yticklabels=labels, cbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    plt.show()

    # Calculate F1 score
    f1 = f1_score(test_y, y_predicted, labels=labels, average='micro')
    print('F1 score micro:', f1)
    f1 = f1_score(test_y, y_predicted, labels=labels, average='weighted')
    print('F1 score weighted:', f1)

# ------------------------------------------------------------ MLP Neural Network ------------------------------------------------------------ #

    # from keras.models import Sequential, load_model, save_model
    # from keras.layers import Dense, BatchNormalization, Dropout, Activation
    # from keras.callbacks import ModelCheckpoint, TensorBoard
    # from keras.initializers import he_uniform, glorot_uniform
    # from keras.constraints import maxnorm
    # from keras.optimizers import Adam
    #
    # # Set random seed of all possible random generated values (for network reproducibility)
    # set_seed()
    #
    # # Split off 10% of the data for model validation
    # X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.1, random_state=157, stratify=data_y.values)
    #
    # # One-Hot encoding of output labels
    # binarizer = LabelBinarizer()
    # y_train = pd.DataFrame(binarizer.fit_transform(y_train.values), columns=['6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95'])
    # y_test = pd.DataFrame(binarizer.fit_transform(y_test.values), columns=['6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95'])
    #
    # # Setup of hyperparameters
    # nb_epoch = 1000
    # batch_size = 256
    #
    # input_dim = X_train.shape[1]
    # output_dim = 14
    # layer_size = 50
    #
    # # init_relu = he_uniform(seed=157)        # used for ReLU
    # init_tanh = glorot_uniform(seed=157)    # used for tanh and softmax
    #
    # # Setup optimization function
    # opt = Adam(learning_rate=0.001)
    #
    # # Neural network starts
    # nn = Sequential()
    #
    # # Input and first hidden layer
    # nn.add(Dense(layer_size, input_dim=input_dim, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    # nn.add(Activation('tanh'))
    # nn.add(BatchNormalization(momentum=0.8))
    # nn.add(Dropout(0.2, seed=157))
    #
    # # Second hidden layer
    # nn.add(Dense(int(layer_size), kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    # nn.add(Activation('tanh'))
    # nn.add(BatchNormalization(momentum=0.8))
    # nn.add(Dropout(0.2, seed=157))
    #
    # # Third hidden layer
    # nn.add(Dense(int(layer_size), kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    # nn.add(Activation('tanh'))
    # nn.add(BatchNormalization(momentum=0.8))
    # nn.add(Dropout(0.1))
    #
    # # Output layer
    # nn.add(Dense(output_dim, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    # nn.add(Activation('softmax'))
    #
    # nn.compile(optimizer=opt,
    #            loss='mean_squared_logarithmic_error',
    #            metrics=['categorical_accuracy'])
    #
    # tensorboard = TensorBoard(log_dir='./logs',
    #                           histogram_freq=0,
    #                           write_graph=True,
    #                           write_images=True)
    #
    # # Training
    # history = nn.fit(x=X_train, y=y_train,
    #                  validation_data=[X_test, y_test],
    #                  epochs=nb_epoch,
    #                  batch_size=batch_size,
    #                  shuffle=True,
    #                  verbose=1,
    #                  callbacks=[tensorboard]).history
    #
    # # Save the model architecture and weights
    # save_model(nn, 'neural_network.h5')
    #
    # # Predict test labels
    # y_predicted = nn.predict_proba(test_X, batch_size=batch_size)
    #
    # # Calculate loss
    # log_loss = multi_weighted_logloss(test_y, y_predicted)
    # print('Multi weighted log loss:', log_loss)
    #
    # # Model training/validation loss plot
    # plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    #
    # # Model training/validation accuracy plot
    # plt.plot(history['categorical_accuracy'])
    # plt.plot(history['val_categorical_accuracy'])
    # plt.title('Categorical Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    # # ----- Calculate confusion matrix and F1 score (after training is done!) ----- #
    #
    # # Load already trained network
    # nn = load_model('neural_network.h5')
    #
    # # Predict test labels
    # y_predicted = nn.predict_proba(test_X, batch_size=batch_size)
    #
    # # Calculate loss
    # log_loss = multi_weighted_logloss(test_y, y_predicted)
    # print('Multi weighted log loss:', log_loss)
    #
    # # Change generated labels for labels given in dataset
    # y_predicted = nn.predict_classes(test_X, batch_size=batch_size)
    #
    # labelizer = {0: 6, 1: 15, 2: 16, 3: 42, 4: 52, 5: 53, 6: 62, 7: 64, 8: 65, 9: 67, 10: 88, 11: 90, 12: 92, 13: 95}
    # y_predicted = np.asarray([labelizer[label] for label in y_predicted])
    #
    # # Confusion matrix plot
    # cf_matrix = confusion_matrix(test_y, y_predicted)
    #
    # labels = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    # ax = sns.heatmap(cf_matrix, annot=True, xticklabels=labels, yticklabels=labels, cbar=False, cmap='Blues')
    # ax.set_title('Confusion Matrix')
    # ax.set_xlabel('Predicted label')
    # ax.set_ylabel('True label')
    # plt.show()
    #
    # # Calculate F1 score
    # f1 = f1_score(test_y, y_predicted, labels=labels, average='weighted')
    # print('F1 score:', f1)
    # f1 = f1_score(test_y, y_predicted, labels=labels, average='weighted')
    # print('F1 score weighted:', f1)

# ------------------------------------------------------------ Code for test set given in Kaggle (unused and obsolete) ------------------------------------------------------------ #

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
